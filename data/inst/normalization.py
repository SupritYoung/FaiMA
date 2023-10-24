import json
import re

def parse_output(s):
    groups = re.findall(r'\[(.*?)\]', s)
    elements = [re.split(r',\s*', group) for group in groups]
    return elements

json_path = './SA-LLM/data/inst/MEMD/AS/train_org.json'
with open(json_path, 'r', encoding='utf-8') as f:
    datas = json.load(f)

for data in datas:
    input_sentence = data['input']
    # matches = re.findall(r"[a-zA-Z]\?", input_sentence)
    # if matches:
    #     for match in matches:
    #         data['input'] = input_sentence.replace(match, match.replace('?', ' ?'))
    #         print(data['input'])
    
    output = parse_output(data['output'])

    for o in output:
        if len(o) > 2:
            print(f'Error output in {input_sentence}')

    # input_list = input_sentence.split()
    # for o in output:
    #     is_in = True
    #     for word in o[0].split():
    #         if word not in input_list:
    #             for w in word.split():
    #                 is_in = False
    #                 break
    #     if not is_in:
    #         print(f'Error input: {input_sentence}')

# 保存更新文件
# with open(json_path, 'w', encoding='utf-8') as f:
#     json.dump(datas, f, indent=4)