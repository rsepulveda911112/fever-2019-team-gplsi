
## This repository was constructed by team GPLSI for the [FEVER shared task 2](http://fever.ai/task.html). 

GPLSI (Natural Language Processing and Information Systems Group) is a research group of Alicante University.
This system has modified the Athene team system for the Fever shared task 1. Team Athene repository is https://github.com/UKPLab/fever-2018-team-athene


### Requirements
* Python 3.6
* AllenNLP
* TensorFlow

### Installation 

* Create a Python Environment and activate it:
```bash 
    virtualenv fever --python=python3
    cd ./fever
    source bin/activate
```
* Install the required dependencies
```bash
    pip install Cython==0.28.5
    pip install numpy==1.14.5 
    pip install -r myrequirements.txt
```
* Download NLTK Punkt Tokenizer
```bash
    python -c "import nltk; nltk.download('punkt')"
```
* Proceed with downloading the data set, the embeddings, the models and the evidence data

### Download the FEVER data set
Download the FEVER dataset from [the website of the FEVER share task](https://sheffieldnlp.github.io/fever/data.html) into the data directory

    mkdir data
    mkdir data/fever-data
    
    #To replicate the paper, download paper_dev and paper_test files. These are concatenated for the shared task
    wget -O data/fever-data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
    wget -O data/fever-data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_dev.jsonl
    wget -O data/fever-data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/shared_task_test.jsonl


### Download the word embeddings

Download pretrained GloVe Vectors

    wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
    unzip glove.6B.zip -d data/glove
    gzip data/glove/*.txt
    
Download pretrained Wiki FastText Vectors

    wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
    mkdir -p data/fasttext
    unzip wiki.en.zip -d data/fasttext


### Download evidence data
The data preparation consists of three steps: (1) downloading the articles from Wikipedia, (2) indexing these for the evidence retrieval and (3) performing the negative sampling for training . 

#### 1. Download Wikipedia data:

Download the pre-processed Wikipedia articles and unzip it into the data folder.
    
    wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
    unzip wiki-pages.zip -d data
 

#### 2. Indexing 
Construct an SQLite Database (go grab a coffee while this runs)

    PYTHONPATH=src python src/scripts/build_db.py data/wiki-pages data/fever/fever.db

 
### Download the GPLSI models
The pre-trained sentence selection ESIM model and the pre-trained claim verification ESIM models. Download the files as followings:

    download this link: https://drive.google.com/file/d/1n0nJO5hPPtLvzp8eOPxHBnQEwlSGSfsq/view?usp=sharing
    
    and unzip models.zip in the root
    
    unzip models.zip
    
### Run the end-to-end pipeline of the submitted models

    PYTHONPATH=src python src/scripts/fever_gplsi/pipeline.py
    
### Run the pipeline in different modes:
Launch the pipeline with optional mode arguments:

    PYTHONPATH=src python src/scripts/fever_gplsi/pipeline.py [--mode <mode>]

All possible modes are as followings:

|Modes|Description|
|---|---|
| `PIPELINE` | Default option. Run the complete pipeline with both training and predicting phases. |
| `PIPELINE_NO_DOC_RETR` | Skip the document retrieval sub-task. With both training and predicting phases. In this case, the datasets already processed by document retrieval are needed.|
| `PIPELINE_RTE_ONLY` | Train and predict only for the RTE sub-task. In this case, the datasets already processed by sentence retrieval are needed.|
| `PREDICT` | Run the all 3 sub-tasks, but only with predicting phase in sentence retrieval and RTE. Only the test set is processed by the document retrieval and sentence retrieval sub-tasks.|
| `PREDICT_NO_DOC_RETR` | Skip the document retrieval sub-task, and only with predicting phase. The test set processed by the document retrieval is needed, and only the test set is processed by sentence retrieval sub-tasks.|
| `PREDICT_RTE_ONLY` | Predict for only the RTE sub-task.|
| `PREDICT_ALL_DATASETS` | Run the all 3 sub-tasks, but only with predicting phase in sentence retrieval and RTE. All 3  datasets are processed by the document retrieval and sentence retrieval sub-tasks.|
| `PREDICT_NO_DOC_RETR_ALL_DATASETS` | Skip the document retrieval sub-task, and only with predicting phase. All 3 datasets processed by the document retrieval are needed. All 3 datasets are processed by sentence retrieval sub-tasks.|
    

### Run the variation of the RTE model
Another variation of the ESIM model is configured through the config file in the conf folder.

To run the models:
    
    PYTHONPATH=src python src/scripts/fever_gplsi/pipeline.py --config conf/<config_file> [--mode <mode>]


### Description of the Config File
The config file regarding the file paths and the hyper parameters is [src/athene/utils/config.py](https://github.com/rsepulveda911112/fever-2019-team-gplsi/blob/master/src/athene/utils/config.py). The descriptions of each field are followings:

|Field|Description|
|---|---|
|model_name|Name of the RTE model. Used as part of the path to save the trained RTE model.|
|glove_path|Path to the pre-trained GloVe word embedding. Either point to the glove.6B.300d.txt.gz or the glove.6B.300d.txt file.|
|fasttext_path|Path to the pre-trained FastText word embedding. Should point to the wiki.en.bin file.|
|ckpt_folder|Path to the checkpoint folder for the trained RTE model. Default as model/<model_name>/rte_checkpoints.|
|db_path|Path to the FEVER database file.|
|dataset_folder|Path to the dataset folder.|
|raw_training_set|Path to the original training set file.|
|raw_dev_set|Path to the original development set file.|
|raw_test_set|Path to the original test set file.|
|training_doc_file|Path to the training set with predicted pages, i.e. the output of the training set through document retrieval sub-task.|
|dev_doc_file|Path to the development set with predicted pages, i.e. the output of the development set through document retrieval sub-task.|
|test_doc_file|Path to the test set with predicted pages, i.e. the output of the test set through document retrieval sub-task.|
|training_set_file|Path to the training set with predicted evidences, i.e. the output of the training set through sentence retrieval sub-task.|
|dev_set_file|Path to the development set with predicted evidences, i.e. the output of the development set through sentence retrieval sub-task.|
|test_set_file|Path to the test set with predicted evidences, i.e. the output of the test set through sentence retrieval sub-task.|
|document_k_wiki|The maximal number of candidate pages for each claim in the document retrieval sub-task.|
|document_parallel|Whether to perform the document retrieval sub-task parallel. True or False.|
|document_add_claim|Whether to append the original claim to the query to the MediaWiki API in the document retrieval sub-task. True or False.|
|submission_file|Path to the final submission file.|
|estimator_name|The name of the RTE estimator referring to [src/athene/rte/utils/estimator_definitions.py](https://github.com/rsepulveda911112/fever-2019-team-gplsi/blob/master/src/athene/rte/utils/estimator_definitions.py).|
|max_sentences|The maximal number of predicted evidences for each claim.|
|max_sentence_size|The maximal length of each predicted evidence. The words that exceed the maximal length are truncated.|
|max_claim_size|The maximal length of each claim. The words that exceed the maximal length are truncated.|
|seed|Random seed of the RTE sub-task.|
|name|The prefix of the checkpoint files for the RTE sub-task. The checkpoint files will be saved in the <ckpt_folder>.|

'**esim_hyper_param**' field contains the hyper parameters regarding the ESIM based model in the RTE sub-task. The descriptions of several special parameters are followings:

|Field|Description|
|---|---|
|num_neurons|The number of neurons for each layer in the model. The first 2 numbers refer to the numbers of neurons of the two bidirectional RNNs in the ESIM model.|
|pos_weight|The positive weights of the 3 classes for the weighted loss. The order is Supported, Refuted, Not Enough Info.|
|max_checks_no_progress|Early stopping policy. Stop training if no improvement in the last x epochs.|
|trainable|Whether to fine tune the word embeddings. True or False.|

'**sentence_retrieval_ensemble_param**' field contains the hyper parameters regarding the ESIM based model in the sentence retrieval sub-task. The descriptions of several special parameters are followings:

|Field|Description|
|---|---|
|num_model|The number of models to ensemble.|
|tf_random_state|The random seeds for the models to ensemble.|
|num_negatives|The number of negative sampling, i.e. false evidences, for each claim in the training phase.|
|c_max_length|The maximal length of each claim. The words that exceed the maximal length are truncated.|
|s_max_length|The maximal length of each candidate evidence sentence. The words that exceed the maximal length are truncated.|
|reserve_embed|Whether to reserve slots in the word embeddings for unseen words. True or False.|
|model_path|Path to the folder for the checkpoint files of the ensemble models.|

Configurations can be exported into json files. To export the current config set, run the script:

    PYTHONPATH=src python src/scripts/fever_gplsi/export_current_config_to_json.py <path/to/output/json>

To use exported configurations, launch the pipeline with argument:

    PYTHONPATH=src python src/scripts/fever_gplsi/pipeline.py --config <path/to/output/json>

### Docker images:
In the docker file is the instructions for building the docker image. You only modify the line for copy the fever database.\
RUN mkdir -p data/fever && wget -O data/fever/fever.db 'http://192.168.100.16:8080/fever.db' 

### Contacts:
If you have any questions please contact the authors.
  * Aimée Alonso Reina aimeear1993@gmail.com 
  * Robiert Sepúlveda Torres rsepulveda911112@gmail.com  
 
### License:
  * Apache License Version 2.0 