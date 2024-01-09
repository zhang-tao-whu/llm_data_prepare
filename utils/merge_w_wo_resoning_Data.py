import json
import os

resoning_datas = '../v1_toy_data/reasoning_datas_v1'
no_resoning_datas = '../v1_toy_data/no_reasoning_datas_v1'
resoning_repeats = 50

collected_Datas = []

resoning_datas = [os.path.join(resoning_datas, file) for file in os.listdir(resoning_datas)]
no_resoning_datas = [os.path.join(no_resoning_datas, file) for file in os.listdir(no_resoning_datas)]

print("start add no reasoning datas.")
for file in no_resoning_datas:
    with open(file, 'r') as f:
        data = json.load(f)
    collected_Datas.extend(data)

print("finished add no reasoning datas.")

print("start add reasoning datas.")
for i in range(resoning_repeats):
    for file in resoning_datas:
        with open(file, 'r') as f:
            data = json.load(f)
        collected_Datas.extend(data)
print("finished add reasoning datas.")

with open('../v1_toy_data/toy_instruction_datas.json', 'w') as f:
    json.dump(collected_Datas, f)