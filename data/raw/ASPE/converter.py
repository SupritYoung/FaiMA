import random
import os
import ast
import json

# 本文件只负责构建测试集、验证集。不负责构建指令数据

os.chdir("./SA-LLM/data/raw/ASPE")

domain_dict = {'14lap':'laptop', '14res':'restaurant', '14twitter':'twitter', 'books':'books', 'clothing':'clothing', 
               'device':'device', 'financial':'finance', 'hotel':'hotel', 'mets-cov':'COVID', 'service': 'service'}

pos_idx_ls, neg_idx_ls, sample_idx_ls = [], [], []
random.seed(33)

num_shot_dict = {'train': 1200, 'dev': 300, 'test': 500}

def read_line_examples_from_dataset_file(data_path, silence=True, num_shot=None):
    sents, labels = [], []
    with open(data_path, 'r', encoding='UTF-8') as fp:
        lines = fp.readlines()
        # 是否采样
        if num_shot and num_shot < len(lines):
            sample_idx_ls = random.sample(range(len(lines)), num_shot)
            lines = [lines[idx] for idx in sample_idx_ls]
        for line in lines:
            line = line.strip()
            text, label_list = line.split("####")
            sents.append(text)
            labels.append(label_list)
    if silence:
        print(f"{data_path}: Total examples = {len(sents)}")
    return sents, labels


def format_list_to_str(lst):
    # 如果不是嵌套列表，则转换为嵌套列表
    res = ""
    if isinstance(lst[0], list):
        res = ", ".join("[" + ", ".join(map(str, l))  + "]" for l in lst)
    else:
        res = "[" + ", ".join(map(str, lst)) + "]"
        # print(res)
    return res

def convert_merge(datasets, files, json_path, txt_path):
    inst_data, txt_data = [], []
    for dataset in datasets:
        for split in files:
            cur_file_path = f"./{dataset}/{split}.txt"    
            # 判断该路径是否存在
            if not os.path.exists(cur_file_path):     
                continue               
            
            if split=='train' and (dataset == '14twitter' or dataset == 'service'):
                # 这两个数据集没有验证集
                num_shot = 1500
            else:
                num_shot = num_shot_dict[split]

            sents, labels = read_line_examples_from_dataset_file(cur_file_path, silence=True, num_shot=num_shot)

            for idx, (sent, label) in enumerate(zip(sents, labels)):
                domain = domain_dict[dataset]
                # 标准化输出的格式
                if dataset == '14twitter':
                    # 这个数据集比较特殊
                    label_list = [l.replace(", ", "").replace(": ", "").replace("-- ", "").replace("'' ", "").replace("- ", "").replace("& ", "").replace("' ", "").replace("/ ", "").replace("| ", "").replace("* ", " ")
                                   for l in ast.literal_eval(label)]
                    txt_data.append(f"{sent}####{label_list}")
                    str_label = format_list_to_str(label_list)
                else:
                    txt_data.append(f"{sent}####{label}")
                    str_label = format_list_to_str(ast.literal_eval(label))
                

                inst_data.append({"id": f'{dataset}_{split}_{idx}',"instruction": "", "input": sent, "output": str_label, 'domain': domain, 'task': 'ASPE'})

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(inst_data, f, ensure_ascii=False, indent=4)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(txt_data))

if __name__ == '__main__':
    datasets = ['14twitter', '14lap', '14res', 'books', 'clothing', 'device', 'financial', 'hotel', 'service']

    json_path = "../../inst/ASPE/train_org.json"
    txt_path = "train_merge.txt"

    convert_merge(datasets, ['train', 'dev'],json_path, txt_path)


    json_path = "../../inst/ASPE/test_org.json"
    txt_path = "test_merge.txt"

    convert_merge(datasets, ['test'], json_path, txt_path)