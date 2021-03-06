# CSE538-final project

This repository contains the final project code for CSE538: Natrual language processing. Our topic is "Conversational Recommendation Systemfor Movies Recommendations". This repo is modified on two wonderful git repos. The first is the offical implementation of the 2018 NeurIPS paper "Towards Deep Conversational Recommendations" [1] https://github.com/RaymondLi0/conversational-recommendations. The second is the official implementation of 2019 EMNLP paper "Towards Knowledge-Based Recommender Dialog System" [2]. https://github.com/THUDM/KBRD.

We upgrade to code of the NeurIPS one to python 3 and torch 1.x version and modify the models of the baseline to align the setting of the other EMNLP paper.

## Requirements

- Python=3.6
- PyTorch >= 1.2
- tqdm
- nltk
- h5py
- numpy
- scikit-learn

## Usage
The repo could be split into two part. One is the baseline part the other is the baseline plus transformer and knowledge graph.  To get the baseline result you should follow the below steps:

### Get the data
Get ReDial data from https://github.com/ReDialData/website/tree/data and Movielens data https://grouplens.org/datasets/movielens/latest/. Note that for the paper we retrieved the Movielens
data set in September 2017. The Movielens latest dataset has been updated since then.
```
git clone https://github.com/silent-chen/CRS.git
cd conversational-recommendations
pip install -r requirements.txt
python -m nltk.downloader punkt

mkdir -p redial movielens
wget -O redial/redial_dataset.zip https://github.com/ReDialData/website/raw/data/redial_dataset.zip
wget -O movielens/ml-latest.zip http://files.grouplens.org/datasets/movielens/ml-latest.zip
# split ReDial data
python scripts/split-redial.py redial/
mv redial/test_data.jsonl redial/test_data
# split Movielens data
python scripts/split-movielens.py movielens/
```

Merge the movie lists by matching the movie names from ReDial and Movielens. Note that this will create an intermediate file `movies_matched.csv`, which is deleted at the end of the script.
```
python scripts/match_movies.py --redial_movies_path=redial/movies_with_mentions.csv --ml_movies_path=movielens/ml-latest/movies.csv --destination=redial/movies_merged.csv
```

### Specify the paths

In the `config.py` file, specify the different paths to use:

- Model weights will be saved in folder `MODELS_PATH='/path/to/models'`
- ReDial data in folder `REDIAL_DATA_PATH='/path/to/redial'`.
This folder must contain three files called `train_data`, `valid_data` and `test_data`
- Movielens data in folder `ML_DATA_PATH='/path/to/movielens'`.
This folder must contain three files called `train_ratings`, `valid_ratings` and `test_ratings`

### Get GenSen pre-trained models

Get GenSen pre-trained models from https://github.com/Maluuba/gensen.
More precisely, you will need the embeddings in the `/path/to/models/embeddings` folder, and 
the following model files: `nli_large_vocab.pkl`, `nli_large.model` in the `/path/to/models/GenSen` folder
```
cd /path/to/models
mkdir GenSen embeddings
wget -O GenSen/nli_large_vocab.pkl https://genseniclr2018.blob.core.windows.net/models/nli_large_vocab.pkl
wget -O GenSen/nli_large.model https://genseniclr2018.blob.core.windows.net/models/nli_large.model
cd embeddings
wget https://raw.githubusercontent.com/Maluuba/gensen/master/data/embedding/glove2h5.py
wget https://github.com/Maluuba/gensen/raw/master/data/embedding/glove2h5.sh
sh glove2h5.sh
cd /path/to/project_dir
```

### Train models

- Train sentiment analysis. This will train a model to predict the movie form labels from ReDial.
The model will be saved in the `/path/to/models/sentiment_analysis` folder
```
python train_sentiment_analysis.py
```
- Train autoencoder recommender system. This will pre-train an Autoencoder Recommender system on Movielens, then fine-tune it on ReDial.
The model will be saved in the `/path/to/models/autorec` folder 
```
python train_autorec.py
```
- Train conversational recommendation model. This will train the whole conversational recommendation model, using the previously trained models.
 The model will be saved in the `/path/to/models/recommender` folder.
```
python train_recommender.py
```
- Train sentiment analysis using transformer. This will train a model to predict the movie form labels from ReDial.
The model will be saved in the `/path/to/models/transformer_sentiment_analysis` folder
```
python train_transformer_sentiment_analysis.py
```
- Train conversational recommendation model using transformer. This will train the whole conversational recommendation model, using the previously trained models.
 The model will be saved in the `/path/to/models/transformer_recommender` folder.
```
python train_transformer_recommender.py
```
### Generate sentences
`generate_responses.py` loads a trained model. 
It takes real dialogues from the ReDial dataset and lets the model generate responses whenever the human recommender speaks
(responses are conditioned on the current dialogue history).
```
python generate_responses.py --model_path=/path/to/models/recommender/model_best --save_path=generations
```

### Knowledge graph

We adopt the model implemented by "Towards Knowledge-Based Recommender Dialog System"[2]. This repo provide a pipeline, which could train the baseline model plus the knowledge graph method.

For more details, pls refer to  https://github.com/THUDM/KBRD.

### Our contributions
1. We added 'train_transformer_sentiment_analysis.py', 'train_transformer_recommender.py', 'models/transformer_sentiment_analysis.py', 'models/transformer_recommender_model.py' and 'models/transformer.py' in order to train models using transformer.
2. We modified the code to adapt to updated Pytorch version.
3. We modified 'config.py' and 'test_params.py' to adjust some parameters.
4. We reproduced the results 'Towards Knowledge-Based Recommender Dialog System' to compare baseline with knowledge-based recommender system.

## Reference
```
[1]
@article{li2018towards,
  title={Towards deep conversational recommendations},
  author={Li, Raymond and Ebrahimi Kahou, Samira and Schulz, Hannes and Michalski, Vincent and Charlin, Laurent and Pal, Chris},
  journal={Advances in neural information processing systems},
  volume={31},
  pages={9725--9735},
  year={2018}
}

[2]
@inproceedings{chen2019towards,
  title={Towards Knowledge-Based Recommender Dialog System},
  author={Chen, Qibin and Lin, Junyang and Zhang, Yichang and Ding, Ming and Cen, Yukuo and Yang, Hongxia and Tang, Jie},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={1803--1813},
  year={2019}
}
```
