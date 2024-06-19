import torch
import os
import ast
import numpy as np

file_path = './pilot_models_results/'
file_list = os.listdir(file_path)
K = 5
file_dict = {}
model_name = []
results_name = []
for i in range(K):
    for name in file_list:
        if str(i) in name and 'model' in name:
            model_name.append(name)
        if str(i) in name and 'results' in name:
            results_name.append(name)


results = []
for result in results_name:
    with open('./pilot_models_results/' + result, 'r') as f:
        data = f.readlines()
        index = len(data)
        for j in range(index):
            if '--- RESULT (pattern_id=0, iteration=0) ---' in data[j]:
                info = data[j + 1]
                break
        info = '{' + info.split('{')[-1]
        # print(info)
        f1 = ast.literal_eval(info)['f1-macro']
        results.append(f1)


results = [np.exp(x)/sum(np.exp(results)) for x in results]
print(results)

parameters = []
for i in range(len(model_name)):
    parameter_meta = {}
    parameter = torch.load('./pilot_models_results/' + model_name[i])
    for j in parameter:
        if 'adapter' in j:
            parameter_meta[j] = parameter[j] * results[i]
    # print(parameter_meta['roberta.encoder.layer.23.output.adapters.bottleneck_adapter.adapter_up.weight'])

    parameters.append(parameter_meta)

distillation_para = parameters[0]
for i in parameters[0]:

    for para in parameters[1:]:
        distillation_para[i] += para[i]

torch.save(distillation_para, './distillation_adapter.bin')






