import json
import torch
import random
from Icl.templates import *
from Icl.ex_retriver import Ex_Retriver
from tqdm import tqdm

domain_dict = {'14lap':'laptop', '14res':'restaurant', '14twitter':'twitter', 'books':'books', 'clothing':'clothing', 
        'device':'device', 'financial':'finance', 'hotel':'hotel', 'mets-cov':'COVID', 'service': 'service'}

def construct_instruct(json_path, save_path, retriever=None, task='ASPE', ICL=True, top_k=5, verbose=False):
    '''
    json_path: 原始，需要建立索引的数据路径
    save_path: 保存的路径
    '''
    sents, labels = [], []
    with open(json_path, 'r', encoding='UTF-8') as fp:
        data = json.load(fp)
        for d in tqdm(data):
            if task == 'ASPE' or task == 'MEMD_AS':
                if ICL:
                    examples = retriever.search_examples(d['input'], top_k=top_k, verbose=verbose)

                    examples_str = ""
                    for id, example in enumerate(examples):
                        examples_str += f'Example {id+1}:\nInput: "{example[0]}"\nOutput: "{example[1]}"\n'

                    prompt = random.choice(aspe_icl_templates).format(example=examples_str, input=d['input'])
                else:
                    prompt = random.choice(aspe_templates).format(input=d['input'])
            elif task == 'MEMD_AOS':
                if ICL:
                    examples = retriever.search_examples(d['input'], top_k=top_k, verbose=verbose)

                    examples_str = ""
                    for id, example in enumerate(examples):
                        examples_str += f'Example {id+1}:\nInput: "{example[0]}"\nOutput: "{example[1]}"\n'

                    prompt = random.choice(aste_icl_templates).format(example=examples_str, input=d['input'])
                else:
                    prompt = random.choice(aste_templates).format(input=d['input'])
            elif task == 'MEMD_ACOS':
                if ICL:
                    examples = retriever.search_examples(d['input'], top_k=top_k, verbose=verbose)

                    examples_str = ""
                    for id, example in enumerate(examples):
                        examples_str += f'Example {id+1}:\nInput: "{example[0]}"\nOutput: "{example[1]}"\n'

                    prompt = random.choice(asqp_icl_templates).format(example=examples_str, input=d['input'])
                else:
                    prompt = random.choice(asqp_templates).format(input=d['input'])
            else:
                raise NotImplementedError

            d['instruction'] = prompt
            d['input'] = ""

    # 直接将数据保存到对应位置
    with open(save_path, 'w', encoding='UTF-8') as fp:
        json.dump(data, fp, indent=4, ensure_ascii=False)

prefix = '/hy-tmp/workspace/SA-LLM/data/inst'
data_files = {
    'ASPE': '/ASPE/',
    'MEMD_AS': '/MEMD/AS/',
    'MEMD_AOS': '/MEMD/AOS/',
    'MEMD_ACOS': '/MEMD/ACOS/',
}

def data_process(task, ICL, paths, method, top_k=5):
    if task not in ['ASPE', 'MEMD_AS', 'MEMD_AOS', 'MEMD_ACOS']:
        raise NotImplementedError

    train_org_path = prefix + data_files[task] + 'train_org.json'
    train_path = prefix + data_files[task] + method + '_train.json'
    test_org_path = prefix + data_files[task] + 'test_org.json'
    test_path = prefix + data_files[task] + method + '_test.json'

    if ICL:
        if method == 'random':
            retriever = Ex_Retriver(ex_file=train_org_path, encode_method='random')
            construct_instruct(train_org_path, train_path, retriever, task=task, ICL=True, top_k=top_k)
            construct_instruct(test_org_path, test_path, retriever, task=task, ICL=True, top_k=top_k)
        elif method == 'sbert':
            retriever = Ex_Retriver(ex_file=train_org_path, paths=paths, encode_method='sbert')
            construct_instruct(train_org_path, train_path, retriever, task=task, ICL=True, top_k=top_k)
            construct_instruct(test_org_path, test_path, retriever, task=task, ICL=True, top_k=top_k)
            # 清除模型释放显存
            del retriever
        elif method == 'gnn':
            retriever = Ex_Retriver(ex_file=train_org_path, paths=paths, encode_method='gnn')
            construct_instruct(train_org_path, train_path, retriever, task=task, ICL=True, top_k=top_k)
            construct_instruct(test_org_path, test_path, retriever, task=task, ICL=True, top_k=top_k, verbose=False)
            del retriever
        else:
            raise NotImplementedError
    else:
        construct_instruct(train_org_path, train_path, task=task, ICL=False)
        construct_instruct(test_org_path, test_path, task=task, ICL=False)

    torch.cuda.empty_cache()

    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    return test_data

# construct_instruct('/hy-tmp/workspace/SA-LLM/data/inst/ASPE/test_org.json', '/hy-tmp/workspace/SA-LLM/data/inst/ASPE/test.json', ICL=False)
