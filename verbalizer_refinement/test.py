import json
import pickle as pkl
import os

origin_path = '../../stance_based_words/argmin/'
path = '../../stance_based_words/words_argmin.json'

def label_info(verbalizer, label_dict):
    label_sta_dict = {}
    for i in verbalizer:
        try:
            label = label_dict[i]
            label_sta_dict[label] = label_sta_dict.setdefault(label, 0) + 1
        except:
            print('Error Key:', i)
    # print(list(label_sta_dict.values()))
    total = sum([int(x) for x in list(label_sta_dict.values())])
    for j in label_sta_dict:
        print('Label: ', j, 'Ratio: ', label_sta_dict[j] / total)



verbalizer_label_dict = {}
verbalizer_topic_dict = {}
verbalizer_label_path = os.listdir(origin_path)
for i in verbalizer_label_path:
    path_meta = i.split('_')
    label = path_meta[0]
    topic = path_meta[1]
    with open(origin_path + i, 'r') as f:
        ver_data = f.readlines()
    ver_data = [x.strip() for x in ver_data]
    for j in ver_data:

        verbalizer_label_dict[j] = label
        verbalizer_topic_dict[j] = topic


with open(path, 'rb') as f:
    data = json.load(f)

for i in data:
    print('Label: ', i)
    print('Number of Verbalizer: ', len(data[i]))
    # print('Verbalizer: ', data[i])
    print('='*20)
    label_info(data[i], verbalizer_topic_dict)
    print('&' * 20)
    label_info(data[i], verbalizer_label_dict)


