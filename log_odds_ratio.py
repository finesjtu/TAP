import math
# from loguru import logger
import tqdm
from loguru import logger
import numpy as np
import pandas as pd
import argparse
import sys
import json
import os

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from text_helper import *

# logger = log.get_logger('root')
#
class LogOddsRatio:
    """
    Log-odds-ratio with informative Dirichlet priors
    """

    def __init__(self, corpus_i, corpus_j, background_corpus=None, lower_case=True, rm_stopwords=True,
                 rm_punctuations=True, tokenizer=None):
        """
        Create a class object and prepare word counts for log-odds-ratio computation
        Args:
            corpus_i:
                A list of documents, each contains a string
            corpus_j:
                A list of documents, each contains a string
            background_corpus (default = None):
                If None, it will be assigned to a concatenation of `corpus_i` and `corpus_j`
            lower_case:
                Whether lower case all words
            rm_stopwords:
                Whether remove stopwords in preprocessing step
            rm_punctuations:
                Whether remove punctuations in preprocessing step
            tokenizer:
                To specify a specific tokenizer for tokenization step
        """

        def preprocessing(corpus):
            if lower_case:
                corpus = [text.lower() for text in corpus]
            logger.info('corpus length: {}'.format(len(corpus)))
            corpus = decontract(corpus)
            tokenized_corpus = parallel_tokenize(corpus, tokenizer)
            if rm_stopwords:
                tokenized_corpus = remove_stopwords(tokenized_corpus)
            if rm_punctuations:
                tokenized_corpus = remove_punctuations(tokenized_corpus)
            return tokenized_corpus

        # Convert a list of string into a list of lists of words
        logger.info("Preprocessing corpus-i")
        corpus_i = preprocessing(corpus_i)
        logger.info("Preprocessing corpus-j")
        corpus_j = preprocessing(corpus_j)
        if background_corpus != None:
            logger.info("Preprocessing corpus-background")
            background_corpus = preprocessing(background_corpus)

        # Compute word counts of every words on each corpus separately
        logger.info("Getting word counts from corpus-i")
        self.y_i = get_word_counts(corpus_i)
        logger.info("Getting word counts from corpus-j")
        self.y_j = get_word_counts(corpus_j)
        logger.info("Getting word counts from corpus-background")
        if background_corpus:
            self.alpha = get_word_counts(background_corpus)
        else:
            # Combine words and sum their counts of corpus i and j in case no specified background corpus
            self.alpha = {k: self.y_i.get(k, 0) + self.y_j.get(k, 0) for k in set(self.y_i) | set(self.y_j)}

        # Sort dicts
        logger.debug("Start sorting and backing up to files")
        self.y_i = {k: v for k, v in sorted(self.y_i.items(), key=lambda item: item[1], reverse=True)}
        self.y_j = {k: v for k, v in sorted(self.y_j.items(), key=lambda item: item[1], reverse=True)}
        self.alpha = {k: v for k, v in sorted(self.alpha.items(), key=lambda item: item[1], reverse=True)}

        # Write to files as backup
        with open("./vocab_back/vocabs_i.txt", "w") as f:
            for k, v in self.y_i.items():
                f.write(f"{k},{v}\n")
        with open("./vocab_back/vocabs_j.txt", "w") as f:
            for k, v in self.y_j.items():
                f.write(f"{k},{v}\n")
        with open("./vocab_back/vocabs_alpha.txt", "w") as f:
            for k, v in self.alpha.items():
                f.write(f"{k},{v}\n")

        # Initialize necessary variables
        self.delta = None
        self.sigma_2 = None
        self.z_scores = None

        # Compute
        logger.info("Start computing delta")
        self._compute_delta()
        logger.info("Start computing sigma^2")
        self._compute_sigma_2()
        logger.info("Start computing Z-score")
        self._compute_z_scores()

        # Sort dicts
        logger.debug("Start sorting and backing up to files")
        self.delta = {k: v for k, v in sorted(self.delta.items(), key=lambda item: item[1], reverse=True)}
        self.sigma_2 = {k: v for k, v in sorted(self.sigma_2.items(), key=lambda item: item[1], reverse=True)}
        self.z_scores = {k: v for k, v in sorted(self.z_scores.items(), key=lambda item: item[1], reverse=True)}

        # Write to files as backup
        with open("./vocab_back/delta.txt", "w") as f:
            for k, v in self.delta.items():
                f.write(f"{k},{v}\n")
        with open("./vocab_back/sigma_2.txt", "w") as f:
            for k, v in self.sigma_2.items():
                f.write(f"{k},{v}\n")
        with open("./vocab_back/z_scores.txt", "w") as f:
            for k, v in self.z_scores.items():
                f.write(f"{k},{v}\n")

    def _compute_delta(self):
        """ The usage difference for word w among two corpora i and j
        """
        self.delta = dict()
        n_i = sum(self.y_i.values())
        n_j = sum(self.y_j.values())
        alpha_zero = sum(self.alpha.values())
        logger.debug(f"Size of corpus-i: {n_i}")
        logger.debug(f"Size of corpus-j: {n_j}")
        logger.debug(f"Size of background corpus: {alpha_zero}")

        try:
            for w in set(self.y_i) | set(self.y_j):  # iterate through all words among two corpora
                first_log = math.log10((self.y_i.get(w, 0) + self.alpha.get(w, 0)) / (
                            n_i + alpha_zero - self.y_i.get(w, 0) - self.alpha.get(w, 0)))
                second_log = math.log10((self.y_j.get(w, 0) + self.alpha.get(w, 0)) / (
                            n_j + alpha_zero - self.y_j.get(w, 0) - self.alpha.get(w, 0)))
                self.delta[w] = first_log - second_log
        except ValueError as e:
            logger.debug(f"Y-i of the word {w}:", self.y_i.get(w, 0))
            logger.debug(f"alpha of the word {w}:", self.alpha.get(w, 0))
            logger.debug(f"value:", (self.y_i.get(w, 0) + self.alpha.get(w, 0)) /
                         (n_i + alpha_zero - self.y_i.get(w, 0) - self.alpha.get(w, 0)))
            raise e

    def _compute_sigma_2(self):
        """ Compute estimated values of sigma squared
        """
        self.sigma_2 = dict()
        for w in self.delta:
            self.sigma_2[w] = (1 / (self.y_i.get(w, 0) + self.alpha.get(w, 0))) + (
                        1 / (self.y_j.get(w, 0) + self.alpha.get(w, 0)))

    def _compute_z_scores(self):
        self.z_scores = dict()
        for w in self.delta:
            self.z_scores[w] = self.delta.get(w, 0) / math.sqrt(self.sigma_2.get(w, 0))


def main():
    # Argument setup
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath_corpus", type=str, required=False,
                        default='/root/NLPCODE/Stance_Detection/Resource/code/mdl-stance-robustness/data/mt_dnn/')
    # parser.add_argument("--filepath_corpus_j", type=str, required=False,
    #                     default='/root/NLPCODE/Stance_Detection/Resource/code/mdl-stance-robustness/data/mt_dnn/')
    parser.add_argument("--dataset", type=str, required=False,
                        default='fnc1', choices=['argmin', 'arc', 'fnc1', 'iac1', 'ibmcs', 'perspectrum', 'semeval2016t6',
                                                   'semeval2019t7', 'snopes'])
    # parser.add_argument("--dataset_j", type=str, required=False,
    #                     default='arc')
    parser.add_argument("--filepath_background_corpus", default=None, type=str, required=False)
    parser.add_argument("--save_path", default='/root/NLPCODE/Stance_Detection/Resource/code/pet/',
                        type=str, required=False)

    parser.add_argument("--save_top_words", default=200, type=int, required=False)
    parser.add_argument("--premise_limit", default=10, type=int, required=False)
    parser.add_argument("--log_level", default=None, type=str, required=False)
    parser.add_argument("--target_fixed", default=True, type=bool, required=False)
    args = parser.parse_args()

    args.filepath_corpus = args.filepath_corpus + args.dataset + '_train.json'
    # args.filepath_corpus_j = args.filepath_corpus_j + args.dataset_j + '_train.json'

    # Set new log level, default is "DEBUG"
    if args.log_level != None:
        logger.remove()
        logger.add(sys.stderr, level=args.log_level)

    premise = []
    label = []
    NOT_PREMISE = False
    with open(args.filepath_corpus, 'r') as corpus_data:
        meta_data = corpus_data.readlines()
        for i in meta_data:
            data = json.loads(i)
            if data['label'] not in label:
                label.append(data['label'])
            if data['premise'] not in premise:
                premise.append(data['premise'])
            if 'hypothesis' not in data.keys():
                NOT_PREMISE = True
                data['hypothesis'] = data['premise']
                # print(data)
    logger.info('Premise list: {}'.format(len(premise)))
    logger.info('Label list: {}'.format(label))
    if len(premise) >= args.premise_limit:
        NOT_PREMISE = True
        premise = ['no_premise']

    for label_ in label:
        for premise_ in premise:
            # Read file into list of texts
            corpus_i = []
            with open(args.filepath_corpus, 'r') as corpus_data:
                meta_data = corpus_data.readlines()
                for i in meta_data:
                    data = json.loads(i)
                    # print(data['premise'])
                    if NOT_PREMISE:
                        if 'hypothesis' not in data.keys():
                            # NOT_PREMISE = True
                            data['hypothesis'] = data['premise']
                        if data['label'] == label_:

                            corpus_i.append(data['hypothesis'])
                    else:
                        if data['label'] == label_ and data['premise'] == premise_:
                            # corpus_i.append(data['premise'])
                            corpus_i.append(data['hypothesis'])
            corpus_j = []
            with open(args.filepath_corpus, 'r') as corpus_data:
                meta_data = corpus_data.readlines()
                for i in meta_data:
                    data = json.loads(i)
                    # print(data['premise'])yhuj89
                    if NOT_PREMISE:
                        if 'hypothesis' not in data.keys():
                            # NOT_PREMISE = True
                            data['hypothesis'] = data['premise']
                        if data['label'] != label_:
                            # corpus_j.append(data['premise'])
                            corpus_j.append(data['hypothesis'])

                    else:
                        if args.target_fixed:
                            if data['label'] != label_ and data['premise'] == premise_:
                                # corpus_j.append(data['premise'])
                                corpus_j.append(data['hypothesis'])
                        else:
                            if data['label'] != label_ or data['premise'] == premise_:
                                # corpus_j.append(data['premise'])
                                corpus_j.append(data['hypothesis'])
            corpus_bg = None

            # Specify the tweet tokenizer
            tweet_tokenizer = TweetTokenizer()
            # Start log-odds-ratio preprocessing
            log_odds_ratio = LogOddsRatio(
                corpus_i, corpus_j, corpus_bg, tokenizer=tweet_tokenizer)

            # Save top words into a file
            if args.save_top_words != None and args.save_top_words > 0:
                if args.save_top_words > len(log_odds_ratio.z_scores):
                    logger.debug('Top words less than vocab size!!')
                    # raise ValueError("--save_top_words must be less than or equal to vocab size")
                save_path = args.save_path + '/stance_based_words/' + args.dataset + '/'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                logger.info(f"Saving top {args.save_top_words} words ranked by Z-score")
                with open(save_path + str(label_) + '_' + premise_ + "_top_words.txt", "w") as f:
                    i = 0
                    for k, v in log_odds_ratio.z_scores.items():
                        f.write(k + "\n")
                        i += 1
                        if i >= args.save_top_words:
                            break


if __name__ == "__main__":
    main()
    logger.success("Done!")