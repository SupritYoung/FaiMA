import json
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from linguistics import SentenceAnalyzer
from utils import compute_sentiment_similarity, compute_domain_similarity, compute_structural_similarity
import warnings

warnings.filterwarnings('ignore')

def split_dev(dataset, dev_ratio):
    """
    将数据集划分为训练集和验证集。
    """
    dev_size = int(len(dataset) * dev_ratio)
    train_size = len(dataset) - dev_size
    return random_split(dataset, [train_size, dev_size])

def parse_output(s):
    groups = re.findall(r'\[(.*?)\]', s)
    elements = [re.split(r',\s*', group) for group in groups]
    return elements


class ABSAGNNDataset(Dataset):
    def __init__(self, file_path, args):
        with open(file_path, 'r') as f:
            self.data = json.load(f)

        self.task = args.task
        self.max_len = args.max_len
        self.lig_threshold = args.lig_threshold
        self.sent_threshold = args.sent_threshold
        # self.struct_threshold = args.struct_threshold
        
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 节点编码器
        self.tokenizer = BertTokenizer.from_pretrained('/hy-tmp/bert-base-uncased')
        self.model = BertModel.from_pretrained('/hy-tmp/bert-base-uncased').to(self.device)
        # 如果有多于一块的 GPU，并行编码
        if torch.cuda.device_count() > 1:  
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.eval()
        # 边编码器
        self.analyzer = SentenceAnalyzer(self.task, threshold=self.lig_threshold)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs = self.data[idx]['input']
        outputs = parse_output(self.data[idx]['output'])
        domains = self.data[idx]['domain']
        return inputs, outputs, domains
    
    def batch_encode_nodes(self, inputs):
        """
        批量编码文本为节点特征
        """
        encoded_inputs = self.tokenizer.batch_encode_plus(inputs, return_tensors='pt', padding='max_length',
                                                          max_length=self.max_len, truncation=True).to(self.device)
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']  # 获取attention_mask

        with torch.no_grad():
            embeddings = self.model(input_ids, attention_mask=attention_mask).last_hidden_state  # 使用attention_mask
        return embeddings


    def batch_generate_edge_features(self, inputs, outputs, domains):
        """
        :return: 边特征，维度为 [b, b, 3]，最后 1 个维度分别为：[语言学相似、领域相似、情感相似]
        """
        # 语言学相似
        lig_features = self.analyzer.linguistic_feature(inputs, outputs)
        # 领域相似
        domain_features = compute_domain_similarity(inputs, domains)
        # 情感相似
        sen_features = compute_sentiment_similarity(inputs, outputs, self.task, threshold=self.sent_threshold)

        # 句子-句子 在各个特征上的 正负例结果可视化
        # 构建 [句子 1， 句子 2， 语言学相似， 领域相似， 情感相似， 结构相似] 的 DataFrame
        # is_visual = True
        # if is_visual:
        #     save_path = f'./results/edge_features/{self.task}_{time.strftime("%Y-%m-%d", time.localtime())}.csv'
        #     try:
        #         df = pd.read_csv(save_path)
        #     except:
        #         df = pd.DataFrame(columns=['s1', 's2', 'linguistic', 'domain', 'sentiment'])
        #     for i in range(len(inputs)):
        #         for j, (l, d, se) in enumerate(zip(lig_features[i], domain_features[i], sen_features[i])):
        #             s1 = inputs[i] + '####' + str(outputs[i])
        #             s2 = inputs[j] + '####' + str(outputs[j])
        #             df.loc[len(df)] = [s1, s2, int(l), int(d), int(se)]
        #     df.to_csv(save_path, index=False)

        # 拼接合并 [b, b] * 4 -> [b, b, 4]
        edge_features = np.stack((lig_features, domain_features, sen_features), axis=-1)

        return torch.from_numpy(edge_features)

    def collate_fn(self, batch):
        '''
        :return node_features: 节点特征，维度为 [b, max_len, dim]
        :return edge_features: 边特征，维度为 [b, b, 4]
        '''
        inputs, outputs, domains = zip(*batch)
        # 获取节点特征
        node_features = self.batch_encode_nodes(inputs)
        # 获取边 对比损失 矩阵
        edge_features = self.batch_generate_edge_features(inputs, outputs, domains)

        return node_features, edge_features


if __name__ == "__main__":
    task = 'ASPE'
    # 必须使用 org 原始数据
    task_path = {'ASPE': './SA-LLM/data/inst/ASPE/train_org.json'}

    train_dataset = ABSAGNNDataset(task_path[task], task=task)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=train_dataset.collate_fn)

    for batch in tqdm(train_loader):
        node_features, edge_features = batch
        # [B,B,4]
        print(node_features.shape)
        print(edge_features.shape)
        # break
