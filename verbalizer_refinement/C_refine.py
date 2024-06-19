import sys
sys.path.append('..')
sys.path.append('../..')
# print(sys.path)
from tasks import load_examples
from pvp import PVPS
from wrapper_refine import TransformerModelWrapper, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig

import json
import numpy as np
import pickle as pkl

PATH = '/root/NLPCODE/Stance_Detection/Resource/code/pet/stance_based_words/'
DATASET = 'semeval2016t6'

DATA_PATH = {
    'argmin':'ArgMin',
    'semeval2016t6':'SemEval2016Task6',
}

SAVE_WORD = 'words_' + DATASET
SAVE_VERBALIZER = 'verbalizer_' + DATASET
SAVE_CANDIDATE = 'candidate_' + DATASET

VERBALIZER_SIZE = 20
DATA_PATH = '../../GLUE_data/'+ DATA_PATH[DATASET] +'/'
TRAIN_SET = "train"


verbalizer_path = PATH + SAVE_WORD + '.json'
# argmin_dict = {
#     '0':'Argument_against',
#     '1':'Argument_for'
# }
# dataset label 不要改
dataset_dict = {
    'argmin':{'0':'Argument_against','1':'Argument_for'},
    'semeval2016t6':{'0':'AGAINST','1':'FAVOR','2':'NONE'},
}

device = 'cuda'
num_examples = -1

def token_2_word(index):
    with open('./vocab.json', 'r') as f:
        token = json.load(f)
    ivert_token = {}
    for key, value in token.items():
        ivert_token[value] = key
    word = []
    for i in index:
        try:
            word.append(ivert_token[i].strip('Ġ'))
        except:
            pass
    return word



# Step1: Calculate the output

# train_data = load_examples(DATASET, DATA_PATH, TRAIN_SET, num_examples=num_examples)
# config = WrapperConfig(model_type='roberta', model_name_or_path='roberta-large', wrapper_type='mlm' , task_name=DATASET, max_seq_length=256,
#                  label_list=list(dataset_dict[DATASET].values()),verbalizer_file = verbalizer_path,label_dict=dataset_dict[DATASET])
# model = TransformerModelWrapper(config)
#
# model.model.to(device)
# outputs, verbalizer_token = model.train(train_data, device, 8)
# outputs = np.array(outputs)
# verbalizer_token = np.array(verbalizer_token)
# # print(verbalizer_token)
# candidate_output = {
#     'token': verbalizer_token,
#     'output': outputs
# }
# with open('./'+SAVE_CANDIDATE+'.pkl', 'wb') as f:
#     pkl.dump(candidate_output, f)



#
# # Step2:c refine
# #
with open('./'+SAVE_CANDIDATE+'.pkl', 'rb') as f:
    data = pkl.load(f)
# print(data)
verbalizer_token_max = data['token']

verbalizer_token_max = np.array(verbalizer_token_max).reshape(-1)
# print(verbalizer_token)
candidate_output = data['output']

candidate_list = []
for i in candidate_output:
    for j in i:
        candidate_list.append(j)

candidate = np.array(candidate_list)
candidate_sum = candidate.sum(axis=0) / len(candidate_list)
candidate_sum = np.array(candidate_sum).reshape(-1)
# print(candidate_sum)
max_index = candidate_sum.argsort()[::-1][:400]
print('*'*40, 'contextualization refine','*'*40)
print(token_2_word(verbalizer_token_max[max_index]))

# Step3: relevance refine
with open('./'+SAVE_CANDIDATE+'.pkl', 'rb') as f:
    data = pkl.load(f)

verbalizer_token = data['token']
# print(verbalizer_token)
candidate_output = data['output']

candidate_list = []
for i in candidate_output:
    for j in i:
        candidate_list.append(j)

candidate = np.array(candidate_list)
samples = candidate.shape[0]
labels = candidate.shape[1]
candidates = candidate.shape[2]
r_relevance = []
for i in range(labels):
    can_rele = []
    for j in range(candidates):
        vec1 = candidate[:,i,0]
        vec2 = candidate[:,i,j]
        r = vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        can_rele.append(r)
    r_relevance.append(can_rele)
print('*'*40, 'relevance refine','*'*40)

rele_index = []
for i in range(labels):
    label_rele = np.array(r_relevance[i])
    label_rele_index = label_rele.argsort()[::-1][:]
    print('label:', i)
    print(label_rele_index)
    print(token_2_word(verbalizer_token[i][label_rele_index]))
    rele_index.append(label_rele_index)




candidate_index = []
for i in range(labels):
    label_candidate_index = []
    for j in verbalizer_token[i][rele_index[i]]:
        if j in verbalizer_token_max[max_index]:
            label_candidate_index.append(j)
    candidate_index.append(label_candidate_index)
for i in range(labels):
    word = token_2_word(list(candidate_index[i]))
    print(word)

print('*'*40, 'verbalizer','*'*40)
verbalizer_data = {}
for i in range(labels):
    word = token_2_word(list(candidate_index[i]))
    verbalizer_data[str(i)] = word[:VERBALIZER_SIZE]

with open(PATH + 'verbalizer/' + SAVE_VERBALIZER + '.json', 'w') as f:
    json.dump(verbalizer_data, f)
print(verbalizer_data)


print(len(r_relevance[0]))