import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

import numpy as np
import os
import math

import config
from models.gensen import GenSenSingle
from utils import sort_for_packed_sequence, neginf

def _normalize(tensor, norm_layer):
    """Broadcast layer norm"""
    size = tensor.size()
    return norm_layer(tensor.view(-1, size[-1])).view(size)
def create_position_codes(n_pos, dim, out):
    position_enc = np.array([
        [pos / np.power(10000, 2 * j / dim) for j in range(dim // 2)]
        for pos in range(n_pos)
    ])

    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc)).type_as(out)
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc)).type_as(out)
    out.detach_()
    out.requires_grad = False

class TransformerEncoder(nn.Module):
    """
    Transformer encoder module.
    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param bool reduction: If true, returns the mean vector for the entire encoding
        sequence.
    :param int n_positions: Size of the position embeddings matrix.
    """
    def __init__(self, embedding_size=1024, ffn_size=1024, n_heads=2, n_layers=2, 
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        padding_idx=0,
        learn_positional_embeddings=False,
        embeddings_scale=False,
        reduction=True,
        n_positions=1024
        ):
        super(TransformerEncoder, self).__init__()

        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.reduction = reduction
        self.padding_idx = padding_idx
        # this is --dropout, not --relu-dropout or --attention-dropout
        self.dropout = nn.Dropout(p=dropout)

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerEncoderLayer(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, tensor, mask):
        """
            input data is a FloatTensor of shape [batch, seq_len, dim]
            mask is a ByteTensor of shape [batch, seq_len], filled with 1 when
            inside the sequence and 0 outside.
        """
        positions = (mask.cumsum(dim=1, dtype=torch.int64) - 1).clamp_(min=0)
        #tensor = self.embeddings(input)
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        # --dropout on the embeddings
        tensor = self.dropout(tensor)

        tensor *= mask.unsqueeze(-1).type_as(tensor)
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)

        if self.reduction:
            divisor = mask.type_as(tensor).sum(dim=1).unsqueeze(-1).clamp(min=1e-7)
            output = tensor.sum(dim=1) / divisor
            return output
        else:
            output = tensor
            return output, mask

class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_heads, embedding_size, ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.attention = MultiHeadAttention(
            n_heads, embedding_size,
            dropout=attention_dropout,  # --attention-dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)
        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tensor, mask):
        tensor = tensor + self.dropout(self.attention(tensor, mask=mask))
        tensor = _normalize(tensor, self.norm1)
        tensor = tensor + self.dropout(self.ffn(tensor))
        tensor = _normalize(tensor, self.norm2)
        tensor *= mask.unsqueeze(-1).type_as(tensor)
        return tensor

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder layer.
    :param int n_heads: the number of multihead attention heads.
    :param int n_layers: number of transformer layers.
    :param int embedding_size: the embedding sizes. Must be a multiple of n_heads.
    :param int ffn_size: the size of the hidden layer in the FFN
    :param embedding: an embedding matrix for the bottom layer of the transformer.
        If none, one is created for this encoder.
    :param float dropout: Dropout used around embeddings and before layer
        layer normalizations. This is used in Vaswani 2017 and works well on
        large datasets.
    :param float attention_dropout: Dropout performed after the multhead attention
        softmax. This is not used in Vaswani 2017.
    :param float relu_attention: Dropout used after the ReLU in the FFN. Not used
        in Vaswani 2017, but used in Tensor2Tensor.
    :param int padding_idx: Reserved padding index in the embeddings matrix.
    :param bool learn_positional_embeddings: If off, sinusoidal embeddings are
        used. If on, position embeddings are learned from scratch.
    :param bool embeddings_scale: Scale embeddings relative to their dimensionality.
        Found useful in fairseq.
    :param int n_positions: Size of the position embeddings matrix.
    """

    def __init__(self, embedding_size=1024, ffn_size=1024, n_heads=2, n_layers=2, 
        embedding=None,
        dropout=0.0,
        attention_dropout=0.0,
        relu_dropout=0.0,
        embeddings_scale=True,
        learn_positional_embeddings=False,
        padding_idx=None,
        n_positions=1024,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dim = embedding_size
        self.embeddings_scale = embeddings_scale
        self.dropout = nn.Dropout(p=dropout)  # --dropout

        self.out_dim = embedding_size
        assert embedding_size % n_heads == 0, \
            'Transformer embedding size must be a multiple of n_heads'

        self.embeddings = embedding

        # create the positional embeddings
        self.position_embeddings = nn.Embedding(n_positions, embedding_size)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, embedding_size, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, embedding_size ** -0.5)

        # build the model
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerDecoderLayer(
                n_heads, embedding_size, ffn_size,
                attention_dropout=attention_dropout,
                relu_dropout=relu_dropout,
                dropout=dropout,
            ))

    def forward(self, input, encoder_output, encoder_mask, incr_state=None):

        seq_len = input.size(1)
        positions = input.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = input
        if self.embeddings_scale:
            tensor = tensor * np.sqrt(self.dim)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)
        tensor = self.dropout(tensor)  # --dropout

        for layer in self.layers:
          tensor = layer(tensor, encoder_output, encoder_mask)
        return tensor, None

class TransformerDecoderLayer(nn.Module):
    def __init__(self, n_heads, embedding_size, ffn_size,
        attention_dropout=0.0,
        relu_dropout=0.0,
        dropout=0.0,
    ):
        super().__init__()
        self.dim = embedding_size
        self.ffn_dim = ffn_size
        self.dropout = nn.Dropout(p=dropout)

        self.self_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm1 = nn.LayerNorm(embedding_size)

        self.encoder_attention = MultiHeadAttention(
            n_heads, embedding_size, dropout=attention_dropout
        )
        self.norm2 = nn.LayerNorm(embedding_size)

        self.ffn = TransformerFFN(embedding_size, ffn_size, relu_dropout=relu_dropout)
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, x, encoder_output, encoder_mask):
        decoder_mask = self._create_selfattn_mask(x)
        # first self attn
        residual = x
        # don't peak into the future!
        x = self.self_attention(query=x, mask=decoder_mask)
        x = self.dropout(x)  # --dropout
        x = x + residual
        x = _normalize(x, self.norm1)

        residual = x
        x = self.encoder_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=encoder_mask
        )
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm2)

        # finally the ffn
        residual = x
        x = self.ffn(x)
        x = self.dropout(x)  # --dropout
        x = residual + x
        x = _normalize(x, self.norm3)

        return x

    def _create_selfattn_mask(self, x):
        # figure out how many timestamps we need
        bsz = x.size(0)
        time = x.size(1)
        # make sure that we don't look into the future
        mask = torch.tril(x.new(time, time).fill_(1))
        # broadcast across batch
        mask = mask.unsqueeze(0).expand(bsz, -1, -1)
        return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, dim, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim

        self.attn_dropout = nn.Dropout(p=dropout)  # --attention-dropout
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.v_lin = nn.Linear(dim, dim)
        # TODO: merge for the initialization step
        nn.init.xavier_normal_(self.q_lin.weight)
        nn.init.xavier_normal_(self.k_lin.weight)
        nn.init.xavier_normal_(self.v_lin.weight)
        # and set biases to 0
        self.out_lin = nn.Linear(dim, dim)

        nn.init.xavier_normal_(self.out_lin.weight)

    def forward(self, query, key=None, value=None, mask=None):
        # Input is [B, query_len, dim]
        # Mask is [B, key_len] (selfattn) or [B, key_len, key_len] (enc attn)
        batch_size, query_len, dim = query.size()
        assert dim == self.dim, \
            f'Dimensions do not match: {dim} query vs {self.dim} configured'
        assert mask is not None, 'Mask is None, please specify a mask'
        n_heads = self.n_heads
        dim_per_head = dim // n_heads
        scale = math.sqrt(dim_per_head)

        def prepare_head(tensor):
            # input is [batch_size, seq_len, n_heads * dim_per_head]
            # output is [batch_size * n_heads, seq_len, dim_per_head]
            bsz, seq_len, _ = tensor.size()
            tensor = tensor.view(batch_size, tensor.size(1), n_heads, dim_per_head)
            tensor = tensor.transpose(1, 2).contiguous().view(
                batch_size * n_heads,
                seq_len,
                dim_per_head
            )
            return tensor

        # q, k, v are the transformed values
        if key is None and value is None:
            # self attention
            key = value = query
        elif value is None:
            # key and value are the same, but query differs
            # self attention
            value = key
        _, key_len, dim = key.size()

        q = prepare_head(self.q_lin(query))
        k = prepare_head(self.k_lin(key))
        v = prepare_head(self.v_lin(value))

        dot_prod = q.div_(scale).bmm(k.transpose(1, 2))
        # [B * n_heads, query_len, key_len]
        attn_mask = (
            (mask == 0)
            .view(batch_size, 1, -1, key_len)
            .repeat(1, n_heads, 1, 1)
            .expand(batch_size, n_heads, query_len, key_len)
            .view(batch_size * n_heads, query_len, key_len)
        )
        assert attn_mask.shape == dot_prod.shape
        dot_prod.masked_fill_(attn_mask, neginf(dot_prod.dtype))

        attn_weights = F.softmax(dot_prod, dim=-1).type_as(query)
        attn_weights = self.attn_dropout(attn_weights)  # --attention-dropout

        attentioned = attn_weights.bmm(v)
        attentioned = (
            attentioned.type_as(query)
            .view(batch_size, n_heads, query_len, dim_per_head)
            .transpose(1, 2).contiguous()
            .view(batch_size, query_len, dim)
        )

        out = self.out_lin(attentioned)

        return out

class TransformerFFN(nn.Module):
    def __init__(self, dim, dim_hidden, relu_dropout=0):
        super(TransformerFFN, self).__init__()
        self.relu_dropout = nn.Dropout(p=relu_dropout)
        self.lin1 = nn.Linear(dim, dim_hidden)
        self.lin2 = nn.Linear(dim_hidden, dim)
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.lin2.weight)
        # TODO: initialize biases to 0

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.relu_dropout(x)  # --relu-dropout
        x = self.lin2(x)
        return x

class HRNN(nn.Module):
    """
    Hierarchical Recurrent Neural Network

    params.keys() ['use_gensen', 'use_movie_occurrences', 'sentence_encoder_hidden_size',
    'conversation_encoder_hidden_size', 'sentence_encoder_num_layers', 'conversation_encoder_num_layers', 'use_dropout',
    ['embedding_dimension']]

    Input: Input["dialogue"] (batch, max_conv_length, max_utterance_length) Long Tensor
           Input["senders"] (batch, max_conv_length) Float Tensor
           Input["lengths"] (batch, max_conv_length) list
           (optional) Input["movie_occurrences"] (batch, max_conv_length, max_utterance_length) for word occurence
                                                 (batch, max_conv_length) for sentence occurrence. Float Tensor
    """
    def __init__(self, params,
                 gensen=False,
                 train_vocabulary=None,
                 train_gensen=True,
                 conv_bidirectional=False):
        super(HRNN, self).__init__()
        self.params = params
        self.use_gensen = bool(gensen)
        self.train_gensen = train_gensen
        self.conv_bidirectional = conv_bidirectional

        self.cuda_available = torch.cuda.is_available()

        # Use instance of gensen if provided
        if isinstance(gensen, GenSenSingle):
            # Assume that vocab expansion is already run on gensen
            self.gensen = gensen
            self.word2id = self.gensen.task_word2id
            # freeze gensen's weights
            if not self.train_gensen:
                for param in self.gensen.parameters():
                    param.requires_grad = False
        # Otherwise instantiate a new gensen module
        elif self.use_gensen:
            self.gensen = GenSenSingle(
                model_folder=os.path.join(config.MODELS_PATH, 'GenSen'),
                filename_prefix='nli_large',
                pretrained_emb=os.path.join(config.MODELS_PATH, 'embeddings/glove.6B.300d.h5'),
                cuda=self.cuda_available
            )
            self.gensen.vocab_expansion(list(train_vocabulary))
            self.word2id = self.gensen.task_word2id
            # freeze gensen's weights
            if not self.train_gensen:
                for param in self.gensen.parameters():
                    param.requires_grad = False
        else:
            self.src_embedding = nn.Embedding(
                num_embeddings=len(train_vocabulary),
                embedding_dim=params['embedding_dimension']
            )
            self.word2id = {word: idx for idx, word in enumerate(train_vocabulary)}
            self.id2word = {idx: word for idx, word in enumerate(train_vocabulary)}
        self.out_features = self.params['sentence_encoder_hidden_size']
        if self.params['use_transformer']:
          self.out_features = self.out_features - 2
          if self.params['use_movie_occurrences'] == "sentence":
            self.out_features = self.out_features - 2

        self.sentence_fc = nn.Linear(
          in_features=2048 + 2*(self.params['use_movie_occurrences'] == "word") if self.use_gensen
          else self.params['embedding_dimension'] + 2*(self.params['use_movie_occurrences'] == "word"),
          out_features=self.out_features
        )
        self.sentence_transformer_encoder = TransformerEncoder(
          embedding_size=self.out_features,
          ffn_size=self.out_features,
          reduction=False
        )
        self.sentence_transformer_decoder = TransformerDecoder(
          embedding_size=self.out_features,
          ffn_size=self.out_features
        )
        self.conversation_transformer_encoder = TransformerEncoder(
          embedding_size=self.params['sentence_encoder_hidden_size'],
          ffn_size=self.params['conversation_encoder_hidden_size'],
          reduction=False
        )
        self.conversation_transformer_decoder = TransformerDecoder(
          embedding_size=self.params['conversation_encoder_hidden_size'],
          ffn_size=self.params['conversation_encoder_hidden_size']
        )
        if self.params['use_dropout']:
            self.dropout = nn.Dropout(p=self.params['use_dropout'])

    def get_sentence_representations(self, dialogue, senders, lengths, movie_occurrences=None):
        batch_size, max_conversation_length = dialogue.data.shape[:2]
        mask = (dialogue != self.word2id['<pad>'])

        #print(dialogue.size())
        # order by descending utterance length
        lengths = lengths.reshape((-1))
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths, self.cuda_available)
        # reshape and reorder
        sorted_utterances = dialogue.view(batch_size * max_conversation_length, -1).index_select(0, sorted_idx)
        sorted_mask = mask.view(batch_size * max_conversation_length, -1).index_select(0, sorted_idx)
        #print('sorted_utterances size is ', sorted_utterances.size())
        # consider sequences of length > 0 only
        num_positive_lengths = np.sum(lengths > 0)
        #print(num_positive_lengths)
        sorted_utterances = sorted_utterances[:num_positive_lengths]
        sorted_mask = sorted_mask[:num_positive_lengths]
        sorted_lengths = sorted_lengths[:num_positive_lengths]

        if self.use_gensen:
            # apply GenSen model and use outputs as word embeddings
            #print("apply GenSen")
            embedded, _ = self.gensen.get_representation_from_ordered(sorted_utterances,
                                                                      lengths=sorted_lengths,
                                                                      pool='last',
                                                                      return_numpy=False)
        else:
            #print("not apply GenSen")
            embedded = self.src_embedding(sorted_utterances)
        # (< batch_size * max conversation_length, max_sentence_length, embedding_size/2048 for gensen)
        # print("EMBEDDED SHAPE", embedded.data.shape)
        #print(embedded.size())
        if self.params['use_dropout']:
            embedded = self.dropout(embedded)

        if self.params['use_movie_occurrences'] == "word":
            if movie_occurrences is None:
                raise ValueError("Please specify movie occurrences")
            # reshape and reorder movie occurrences by utterance length
            movie_occurrences = movie_occurrences.view(
                batch_size * max_conversation_length, -1).index_select(0, sorted_idx)
            # keep indices where sequence_length > 0
            movie_occurrences = movie_occurrences[:num_positive_lengths]
            embedded = torch.cat((embedded, movie_occurrences.unsqueeze(2)), 2)
            if self.params['use_transformer'] == True:
              embedded = torch.cat((embedded, movie_occurrences.unsqueeze(2)), 2)

        #print(embedded.size())
        embedded= self.sentence_fc(embedded)
        sentence_representations, encoder_mask = self.sentence_transformer_encoder(embedded, sorted_mask)
        sentence_representations, _ = self.sentence_transformer_decoder(embedded, sentence_representations, encoder_mask)
        sentence_representations = sentence_representations[:, -1, :]
        #print('sentence_representations size is', sentence_representations.size())
        if self.params['use_dropout']:
            sentence_representations = self.dropout(sentence_representations)

        # Complete the missing sequences (of length 0)
        if num_positive_lengths < batch_size * max_conversation_length:
            tt = torch.cuda.FloatTensor if self.cuda_available else torch.FloatTensor
            pad_tensor = Variable(torch.zeros(
                batch_size * max_conversation_length - num_positive_lengths,
                self.out_features,
                out=tt()
            ))
            sentence_representations = torch.cat((
                sentence_representations,
                pad_tensor
            ), 0)
        
        # print("SENTENCE REP SHAPE",
        #       sentence_representations.data.shape)  # (batch_size * max_conversation_length, 2*hidden_size)
        # Retrieve original sentence order and Reshape to separate conversations
        sentence_representations = sentence_representations.index_select(0, rev).view(
            batch_size,
            max_conversation_length,
            self.out_features)
        # Append sender information
        sentence_representations = torch.cat([sentence_representations, senders.unsqueeze(2)], 2)
        if self.params['use_transformer'] == True:
          sentence_representations = torch.cat([sentence_representations, senders.unsqueeze(2)], 2)
        # Append movie occurrence information if required
        if self.params['use_movie_occurrences'] == "sentence":
            if movie_occurrences is None:
                raise ValueError("Please specify movie occurrences")
            sentence_representations = torch.cat((sentence_representations, movie_occurrences.unsqueeze(2)), 2)
            if self.params['use_transformer'] == True:
              sentence_representations = torch.cat((sentence_representations, movie_occurrences.unsqueeze(2)), 2)
        # print("SENTENCE REP SHAPE WITH SENDER INFO", sentence_representations.data.shape)
        #  (batch_size, max_conv_length, 513 + self.params['use_movie_occurrences'])
        return sentence_representations

    def forward(self, input_dict, return_all=True, return_sentence_representations=False):
        movie_occurrences = input_dict["movie_occurrences"] if self.params['use_movie_occurrences'] else None
        # get sentence representations
        sentence_representations = self.get_sentence_representations(
            input_dict["dialogue"], input_dict["senders"], lengths=input_dict["lengths"],
            movie_occurrences=movie_occurrences)
        # (batch_size, max_conv_length, 2*sent_hidden_size + 1 + use_movie_occurences)
        # Pass whole conversation into GRU
        lengths = input_dict["conversation_lengths"]
        #print(lengths)
        #print(sentence_representations.size())
        representation_sum = torch.sum(sentence_representations, dim=2)
        conv_mask = (representation_sum != 0)
        #print(conv_mask)
        sorted_lengths, sorted_idx, rev = sort_for_packed_sequence(lengths, self.cuda_available)

        # reorder in decreasing sequence length
        sorted_representations = sentence_representations.index_select(0, sorted_idx)

        encoder_representations, encoder_mask = self.conversation_transformer_encoder(sorted_representations, conv_mask)
        conversation_representations, _ = self.conversation_transformer_decoder(sorted_representations, encoder_representations, encoder_mask)

        # retrieve original order
        conversation_representations = conversation_representations.index_select(0, rev)
        # print("LAST STATE SHAPE", last_state.data.shape) # (num_layers * num_directions, batch, conv_hidden_size)
        if self.params['use_dropout']:
            conversation_representations = self.dropout(conversation_representations)

        if return_all:
            if not return_sentence_representations:
                # return the last layer of the GRU for each t.
                # (batch_size, max_conv_length, hidden_size*num_directions
                return conversation_representations
            else:
                # also return sentence representations
                return conversation_representations, sentence_representations
        else:
            return conversation_representations[:, -1, :]

