import argparse
import os
from enum import Enum

from athene.retrieval.document.docment_retrieval import main as document_retrieval_main
from athene.retrieval.sentences.ensemble import entrance as sentence_retrieval_ensemble_entrance
from athene.utils.config import Config
from common.util.log_helper import LogHelper
from scripts.fever_gplsi.rte import entrance as rte_main


class Mode(Enum):
    PIPELINE = 1  # Run the whole pipeline, training & predicting
    PIPELINE_NO_DOC_RETR = 2  # Skip the document retrieval sub-task. Training & predicting
    PIPELINE_RTE_ONLY = 3  # Run only the RTE sub-task. Training & predicting
    PREDICT = 4  # Run all 3 sub-tasks but no training, and only predict test set with pre-trained models of sentence retrieval and RTE.
    PREDICT_NO_DOC_RETR = 5  # Skip the document retrieval sub-task. No training and only predict test set with pre-trained models of sentence retrieval and RTE.
    PREDICT_RTE_ONLY = 6  # Predict test set only for the RTE sub-task with pre-trained model of RTE.
    PREDICT_ALL_DATASETS = 7  # Run all 3 sub-tasks but no training. Predict all 3 datasets for document retrieval and sentence retrieval.
    PREDICT_NO_DOC_RETR_ALL_DATASETS = 8  # Skip the document retrieval sub-task. No training. Predict all datasets for sentence retrieval.

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Mode[s]
        except KeyError:
            raise ValueError()

def _construct_args_for_sentence_retrieval(phase='training'):
    from argparse import Namespace
    _args = Namespace()
    for k, v in Config.sentence_retrieval_ensemble_param.items():
        setattr(_args, k, v)
    setattr(_args, 'train_data', Config.training_doc_file)
    setattr(_args, 'dev_data', Config.dev_doc_file)
    setattr(_args, 'test_data', Config.test_doc_file)
    setattr(_args, 'fasttext_path', Config.fasttext_path)
    setattr(_args, 'phase', phase)
    if phase == 'training':
        out_file = Config.training_set_file
    elif phase == 'deving':
        out_file = Config.dev_set_file
    else:
        out_file = Config.test_set_file
    setattr(_args, 'out_file', out_file)
    return _args


def sentence_retrieval_ensemble(logger, mode: Mode = Mode.PIPELINE):
    logger.info("Starting data pre-processing...")
    tmp_file = os.path.join(Config.dataset_folder, "tmp.jsonl")
    with open(tmp_file, 'w') as wf:
        files = [Config.training_doc_file, Config.dev_doc_file, Config.test_doc_file]
        for f in files:
            with open(f) as rf:
                for line in rf:
                    wf.write(line)
    _args = _construct_args_for_sentence_retrieval()
    _args.phase = 'data'
    _args.test_data = tmp_file
    os.remove(tmp_file)
    if mode in {Mode.PIPELINE, Mode.PIPELINE_NO_DOC_RETR}:
        logger.info("Starting training sentence retrieval...")
        _args.phase = 'training'
        _args.test_data = Config.dev_doc_file  # predict dev set in training phase
        os.remove(os.path.join(os.getcwd(), "test_data.p"))
        os.remove(os.path.join(os.getcwd(), "test_indexes.p"))
        sentence_retrieval_ensemble_entrance(_args)
        logger.info("Finished training sentence retrieval.")
    if mode in {Mode.PIPELINE, Mode.PIPELINE_NO_DOC_RETR, Mode.PREDICT_ALL_DATASETS,
                Mode.PREDICT_NO_DOC_RETR_ALL_DATASETS}:
        logger.info("Starting selecting sentences for dev set...")
        _args.phase = 'testing'
        _args.out_file = Config.dev_set_file
        _args.test_data = Config.dev_doc_file
        os.remove(os.path.join(os.getcwd(), "test_data.p"))
        os.remove(os.path.join(os.getcwd(), "test_indexes.p"))
        sentence_retrieval_ensemble_entrance(_args)
        logger.info("Finished selecting sentences for dev set.")
        logger.info("Starting selecting sentences for training set...")
        os.remove(os.path.join(os.getcwd(), "test_data.p"))
        os.remove(os.path.join(os.getcwd(), "test_indexes.p"))
        _args.test_data = Config.training_doc_file
        _args.phase = 'testing'
        _args.out_file = Config.training_set_file
        sentence_retrieval_ensemble_entrance(_args)
        logger.info("Finished selecting sentences for training set.")
    logger.info("Starting selecting sentences for test set...")
    try:
        os.remove(os.path.join(os.getcwd(), "test_data.p"))
        os.remove(os.path.join(os.getcwd(), "test_indexes.p"))
    except:
        print("")
    _args.test_data = Config.test_doc_file
    _args.phase = 'testing'
    _args.out_file = Config.test_set_file
    sentence_retrieval_ensemble_entrance(_args, calculate_fever_score=False)
    logger.info("Finished selecting sentences for test set.")


def document_retrieval(logger, mode: Mode = Mode.PIPELINE):
    if mode in {Mode.PIPELINE, Mode.PREDICT_ALL_DATASETS}:
        logger.info("Starting document retrieval for training set...")
        document_retrieval_main(Config.db_path(), Config.document_k_wiki, Config.raw_training_set,
                                Config.training_doc_file,
                                Config.document_add_claim, Config.document_parallel)
        logger.info("Finished document retrieval for training set.")
        logger.info("Starting document retrieval for dev set...")
        document_retrieval_main(Config.db_path(), Config.document_k_wiki, Config.raw_dev_set, Config.dev_doc_file,
                                Config.document_add_claim, Config.document_parallel)
        logger.info("Finished document retrieval for dev set.")
    logger.info("Starting document retrieval for test set...")
    document_retrieval_main(Config.db_path(), Config.document_k_wiki, Config.relative_path_test_file, Config.test_doc_file,
                            Config.document_add_claim, Config.document_parallel)
    logger.info("Finished document retrieval for test set.")


def rte(logger, args, mode: Mode = Mode.PIPELINE):
    claim_validation_estimator = None
    if mode in {Mode.PIPELINE_NO_DOC_RETR, Mode.PIPELINE, Mode.PIPELINE_RTE_ONLY}:
        logger.info("Starting training claim validation...")
        claim_validation_estimator = rte_main("train", args.config)
        logger.info("Finished training claim validation.")
    logger.info("Starting testing claim validation...")
    rte_main("test", args.config, claim_validation_estimator)
    logger.info("Finished testing claim validation.")


class NullArgs:
    def __getattr__(self, item):
        return None


def main(args=NullArgs()):
    LogHelper.setup()
    logger = LogHelper.get_logger(os.path.splitext(os.path.basename(__file__))[0])
    # For only classify
    args.mode = Mode.PREDICT
    if args.config is not None:
        Config.load_config(args.config)

    if args.out_file is not None:
        Config.relative_path_submission = args.out_file

    if args.in_file is not None:
        Config.relative_path_test_file = args.in_file

    if args.database is not None:
        Config.relative_path_db = args.database

    print("relative_path_db " + Config.relative_path_db)
    print("raw_test_set " + Config.raw_test_set())

    if os.path.exists(Config.test_doc_file):
        os.remove(Config.test_doc_file)
    if os.path.exists(Config.test_set_file):
        os.remove(Config.test_set_file)

    if args.mode in {Mode.PIPELINE, Mode.PREDICT, Mode.PREDICT_ALL_DATASETS}:
        logger.info(
            "=========================== Sub-task 1. Document Retrieval ==========================================")
        document_retrieval(logger, args.mode)
    if args.mode in {Mode.PIPELINE_NO_DOC_RETR, Mode.PIPELINE, Mode.PREDICT, Mode.PREDICT_NO_DOC_RETR,
                     Mode.PREDICT_ALL_DATASETS, Mode.PREDICT_NO_DOC_RETR_ALL_DATASETS}:
        logger.info(
            "=========================== Sub-task 2. Sentence Retrieval ==========================================")
        sentence_retrieval_ensemble(logger, args.mode)
    logger.info("=========================== Sub-task 3. Claim Validation ============================================")
    rte(logger, args, args.mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str)
    parser.add_argument('--out_file', type=str)
    parser.add_argument('--in_file', type=str)

    parser.add_argument('--config', help='/path/to/config/file, in JSON format')
    parser.add_argument('--mode', type=Mode.from_string, choices=list(Mode), help='mode of the execution',
                        default=Mode.PIPELINE)
    args = parser.parse_args()
    main(args)
