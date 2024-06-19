import csv
from time import time
from collections import Counter
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import os
from vocab import Vocabulary
from collections import defaultdict
import json
import ast

def compute_acc(predicts, labels):
    return accuracy_score(labels, predicts)

def compute_f1(predicts, labels):
    return f1_score(labels, predicts)

def compute_f1_clw(predicts, labels):
    return f1_score(labels, predicts, average=None).tolist()

def compute_precision_clw(predicts, labels):
    return precision_score(labels, predicts, average=None).tolist()

def compute_recall_clw(predicts, labels):
    return recall_score(labels, predicts, average=None).tolist()

def compute_f1_macro(predicts, labels):
    return f1_score(labels, predicts, average="macro")

def compute_f1_micro(predicts, labels):
    return f1_score(labels, predicts, average="micro")

def compute_precision_macro(predicts, labels):
    return precision_score(labels, predicts, average="macro")

def compute_recall_macro(predicts, labels):
    return recall_score(labels, predicts, average="macro")

def compute_mcc(predicts, labels):
    return 100.0 * matthews_corrcoef(labels, predicts)

def compute_pearson(predicts, labels):
    pcof = pearsonr(labels, predicts)[0]
    return 100.0 * pcof

def compute_spearman(predicts, labels):
    scof = spearmanr(labels, predicts)[0]
    return 100.0 * scof



# argmin
ArgMinLabelMapper = Vocabulary(True)
ArgMinLabelMapper.add('Argument_against')
ArgMinLabelMapper.add('Argument_for')


ARCLabelMapper.add('disagree')

# semeval2016 task 6
SemEval2016T6LabelMapper = Vocabulary(True)
SemEval2016T6LabelMapper.add('AGAINST')
SemEval2016T6LabelMapper.add('FAVOR')
SemEval2016T6LabelMapper.add('NONE')


GLOBAL_MAP = {
 'argmin': ArgMinLabelMapper,
 'semeval2016t6': SemEval2016T6LabelMapper,

}

root = 'path'


argmin_path = os.path.join(root, 'ArgMin/')



semeval2016t6_path = os.path.join(root, 'SemEval2016Task6/')



import re
import zipfile
def load_semeval2019t7(file, header=True, create_adversarial=False, dataset=None, MAX_SEQ_LEN=100,
                 OPENNMT_GPU=-1, OPENNMT_BATCH_SIZE=30):

    def parse_tweets(folder_path):
        # create a dict with key = reply_tweet_id and values = source_tweet_id, source_tweet_text, reply_tweet_text
        tweet_dict = {}
        with zipfile.ZipFile(file+folder_path, 'r') as z:
            for filename in z.namelist():
                if not filename.lower().endswith(".json") or filename.rsplit("/", 1)[1] in ['raw.json', 'structure.json', 'dev-key.json', 'train-key.json']:
                    continue
                with z.open(filename) as f:
                    data = f.read()
                    d = json.loads(data.decode("ISO-8859-1"))

                    if "data" in d.keys(): #reddit
                        if "body" in d['data'].keys(): # reply
                            tweet_dict[d['data']['id']] = d['data']['body']
                        elif "children" in d['data'].keys() and isinstance(d['data']['children'][0], dict):
                            tweet_dict[d['data']['children'][0]['data']['id']] = d['data']['children'][0]['data']['title']
                        else: # source
                            try:
                                tweet_dict[d['data']['children'][0]] = ""
                            except Exception as e:
                                print(e)

                    if "text" in d.keys(): # twitter
                        tweet_dict[str(d['id'])] = d['text']

        return tweet_dict

    def read_and_tokenize_json(file_name, tweet_dict):
        X, y = [], []

        with open(file+file_name, "r") as in_f:
            split_dict = json.load(in_f)['subtaskaenglish']

            for tweet_id, label in split_dict.items():
                X_meta = []
                try:
                    X_meta = tweet_dict[tweet_id]
                except:
                    continue
                X_meta.append(label)
                X.append(X_meta)
        return X

    def read_and_tokenize_zip(folder_path, set_file, tweet_dict):
        X, y = [], []

        with zipfile.ZipFile(file + folder_path, 'r') as z:
            with z.open("rumoureval-2019-training-data/"+set_file) as in_f:
                split_dict = json.load(in_f)['subtaskaenglish']

                for tweet_id, label in split_dict.items():
                    X_meta = []
                    try:
                        X_meta.append(tweet_dict[tweet_id])
                    except:
                        continue
                    X_meta.append(label)
                    X.append(X_meta)

            return X

    print("\n===================== Start preprocessing file: "+ file + " =====================")
    t_total = time()

    # read train/dev data
    # tweet_dict = parse_tweets("rumoureval-2019-training-data.zip")
    X = read_and_tokenize_zip(file + "rumoureval-2019-training-data.zip", "train-key.json", parse_tweets("rumoureval-2019-training-data.zip"))
    # X_dev, y_dev = read_and_tokenize_zip("rumoureval-2019-training-data.zip", "dev-key.json", parse_tweets("rumoureval-2019-training-data.zip"))
    #
    # # read test data
    # tweet_dict = parse_tweets("rumoureval-2019-test-data.zip")
    # X_test, y_test = read_and_tokenize_json("final-eval-key.json", parse_tweets("rumoureval-2019-test-data.zip"))
    print(len(X))
    for i in range(10):
        print(X[i])


if __name__ == '__main__':
    # root = '/root/NLPCODE/Stance_Detection/Resource/code/pet/GLUE_data/'
    # semeval2016t6_path = os.path.join(root, 'SemEval2016Task6/')

    load_semeval2019t7(semeval2019t7_path, GLOBAL_MAP['semeval2019t7'], dataset='semeval2019t7', MAX_SEQ_LEN=200)