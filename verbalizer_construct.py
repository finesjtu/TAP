from transformers import RobertaTokenizer
import os
import json

'''
    1、verbalizer_construct.py aims to save candidates from log-odds-ratio.py with removing the duplicate words and un-encode words.
    And dataset_dict is the label word for DATASET, which is the base r_index in C-refine.py.
        output: words_DATASET.json
    2、C_refine.py calculates the contextualization and relevance to refine the candidates.
        output: candidate_DATASET.pkl(dump the prediction probability) and verbalizer_DATASET.json(final verbalizer)
'''



PATH = '/root/NLPCODE/Stance_Detection/Resource/code/pet/stance_based_words/'
# '/root/NLPCODE/Stance_Detection/Resource/code/mdl-stance-robustness/data/mt_dnn/stance_based_words'
DATASET = 'semeval2016t6'
SAVE_WORD = 'words_' + DATASET
KEEP_WORDS = -1

dataset_dict = {

    'argmin':{'0':'against','1':'for'},

    'semeval2016t6':{'0':'against','1':'favor','2':'none'},

}

def remove_duplicate(verbalizer_dict):
    value = []
    del_word = []
    values = list(verbalizer_dict.values())
    for i in values:
        for j in i:
            # print(j)
            if j not in value:
                value.append(j)
            else:
                del_word.append(j)
    for key in verbalizer_dict.keys():

        for z in del_word:
            if z in verbalizer_dict[key]:
                verbalizer_dict[key].remove(z)
    print(del_word)
    return verbalizer_dict

def tokenizer_filter(tokenizer, path):
    with open(path, 'r') as f:
        words = f.readlines()
    words_bak = []
    for word in words:
        word = word.replace('\n', '')
        token = tokenizer.encode(word, add_special_tokens=False, add_prefix_space=True)
        # print(token)
        # print(word)
        if len(token) == 1:
            words_bak.append(word)
    return words_bak



def words_token(words_path, labels=4, keep_words=KEEP_WORDS):

    file_list = os.listdir(words_path)
    labels = []
    for i in file_list:
        label = i.split('_')[0]
        if label not in labels:
            labels.append(label)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    label_dict = {}
    verbalizer = {}
    for i in range(len(labels)):
        label_file = [name for name in file_list if str(i) in name]
        label_dict[i] = label_file

    for i in range(len(labels)):
        file_list = label_dict[i]
        words = []
        for j in file_list:
            path = words_path + '/' + j
            words += tokenizer_filter(tokenizer, path)[:keep_words]

        verbalizer[i] = words
    # print(verbalizer)
    verbalizer_ = remove_duplicate(verbalizer)
    for key, value in verbalizer_.items():
        verbalizer_[key] = [dataset_dict[DATASET][str(key)]] + verbalizer_[key]
    with open(PATH + SAVE_WORD + '.json', 'w') as f:
        json.dump(verbalizer_, f)
    for i in verbalizer_.items():
        print(i[0],i[1][:KEEP_WORDS])





words_token(PATH + DATASET )
#
# dict_1 = {0: ['information', 'immigration', 'cultural', 'thumbs', 'trump', 'accounting', 'publishing'], 1: ['takeover', 'cis', 'prince', 'downgrade', 'stink', 'signal'], 2: ['capital', 'obese', 'writing', 'citizenship', 'polygamy', 'spelling'], 3: ['comments', 'libraries', 'library', 'banks', 'trump', 'gateway']}
#
# values = list(dict_1.values())
# value = []
# del_word = []
# for i in values:
#     for j in i:
#         # print(j)
#         if j not in value:
#             value.append(j)
#         else:
#             del_word.append(j)
# for key in dict_1.keys():
#
#     for z in del_word:
#         if z in dict_1[key]:
#             dict_1[key].remove(z)
# print(dict_1)



