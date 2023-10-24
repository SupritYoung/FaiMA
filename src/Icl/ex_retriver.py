import faiss
import json
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import random
import numpy as np
from tqdm import tqdm

from gnnencoder.encoder import SentenceEncoder

class Ex_Retriver():
    def __init__(self, ex_file, paths=None, encode_method='sbert'):
        '''
        input: ex_file: 需要构建检索的例子文件（一般为原始训练集）
        '''
        self.encode_method = encode_method
        self.top_k = 5

        with open(ex_file, 'r') as f:
            data = json.load(f)
            self.sents = []
            self.labels = []
            for d in data:
                self.sents.append(d['input'])
                self.labels.append(d['output'])
        self.data_dict = {}
        for sent, label in zip(self.sents, self.labels):
            self.data_dict[sent] = label

        # Initialize different models based on the specified method
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if encode_method == 'sbert':
             # 预先初始化所有句子的向量并存储在index中
            self.model = AutoModel.from_pretrained(paths['sbert_path']).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(paths['sbert_path'])
            self.init_embeddings(self.sents)
        elif encode_method == 'bert':
            self.model = AutoModel.from_pretrained(paths['bert_path']).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(paths['bert_path'])
            self.init_embeddings(self.sents)
        elif encode_method == 'gnn':
            # TODO 这里需要修改指定路径
            gnn_model_path = paths['gnn_path']
            # bert_model_path = paths['bert']
            bert_model_path = '/hy-tmp/bert-base-uncased'
            self.gnn_encoder = SentenceEncoder(gnn_model_path, bert_model_path)
            self.init_embeddings(self.sents)
        elif encode_method == 'random':
            pass
        else:
            raise NotImplementedError
        
    def encode_sentences(self, sents, batch_size=512):
        '''
        sents: 所有需要编码的句子
        '''
        if self.encode_method == 'sbert':
            return self.bert_encode_sentences(sents)
        elif self.encode_method == 'bert':
            return self.bert_encode_sentences(sents)
        elif self.encode_method == 'gnn':
            return self.gnn_encode_sentences(sents)
        elif self.encode_method is None:
            return None
        else:
            raise NotImplementedError
        
    def gnn_encode_sentences(self, sents, batch_size=512):
        '''
        sents: 所有需要编码的句子
        '''
        # all_embeddings = []
        # TODO 目前写死成了 维度 = 4 
        all_dims_embeddings = [[], [], [], []]

        for i in range(0, len(sents), batch_size):
            # 每个维度分别构建检索
            batch_sents = sents[i:i + batch_size]
            dims_representations, avg_representation = self.gnn_encoder.encode(batch_sents)
            avg_embeddings = avg_representation.cpu().numpy()

            # embeddings = F.normalize(embeddings, p=2, dim=1)
            # all_embeddings.append(embeddings)
            for i in range(3):
                all_dims_embeddings[i].append(dims_representations[i].cpu().numpy())
            all_dims_embeddings[3].append(avg_embeddings)

        # return np.concatenate(all_embeddings, axis=0)
        return [np.concatenate(all_dims_embeddings[i], axis=0) for i in range(4)]

    def bert_encode_sentences(self, sents, batch_size=512):
        '''
        sents: 所有需要编码的句子
        batch_size: 每次编码的batch size
        '''
        all_embeddings = []

        for i in range(0, len(sents), batch_size):
            batch_sents = sents[i:i + batch_size]
            encoded_input = self.tokenizer(batch_sents, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}  # Move input to device

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            if self.encode_method == 'sbert':
                def mean_pooling(model_output, attention_mask):
                    token_embeddings = model_output[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            elif self.encode_method == 'bert':
                embeddings = model_output[0][:, 0, :]

            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def init_embeddings(self, sents):
        print("Initializing embeddings...")
        # build the index using FAISS
        embeddings = self.encode_sentences(sents)

        if self.encode_method == 'gnn':
            # 针对每个特征维度分别构建检索
            d = embeddings[0].shape[1]
            self.index = [faiss.IndexFlatL2(d) for i in range(5)]
            for i in range(4):
                self.index[i].add(embeddings[i])
        else:
            d = embeddings.shape[1]

            self.index = faiss.IndexFlatL2(d)
            self.index.add(embeddings)

    def search_examples(self, query, top_k=3, verbose=False):
        if self.encode_method == 'random':
            return random.sample(list(zip(self.data_dict.keys(), self.data_dict.values())), top_k)
        
        if verbose:
            print(f"\nSearching for: {query}")

        if top_k is None:
            top_k = self.top_k

        if self.encode_method == 'gnn':
            query_embeddings = self.encode_sentences([query])

            choosed_idxs = {} # 已选择的索引
            # 每个维度要选择的数量
            n_dims = [1, 1, 1, 2]
            feture_types = ['lig', 'domain', 'senti', 'avg']
            for i in range(4):
                distances, indices = self.index[i].search(query_embeddings[i], self.index[i].ntotal)
                # 距离越小的放到前面（这样最相似的例子离输入最近）
                sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1], reverse=False)
                # 每个维度取对应数量，且去重
                for idx, dist in sorted_results:
                    # 去重，同一句子只出现一次
                    if idx not in choosed_idxs.keys() and n_dims[i] > 0 and self.sents[idx] != query:
                        choosed_idxs[idx] = dist
                        n_dims[i] -= 1
                        
                        if verbose:
                            print(f'{feture_types[i]}: {self.sents[idx]}')
                    if n_dims[i] == 0:
                        break
            # 将字典转换为列表，然后按照距离排序，距离大的放到前面，距离小的放到后面（离输入更近）
            choosed_idxs = sorted(choosed_idxs.items(), key=lambda x: x[1], reverse=True)
            res = [(self.sents[idx], self.data_dict[self.sents[idx]]) for idx, dist in choosed_idxs]
            return res
        else:
            query_embedding = self.encode_sentences([query])
            distances, indices = self.index.search(query_embedding, self.index.ntotal)

            # 距离越大的放到前面（这样最相似的例子离输入最近）
            sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1], reverse=True)
            # Getting the top k results
            top_results = sorted_results[:top_k]
            res = [(self.sents[idx], self.data_dict[self.sents[idx]]) for idx, dist in top_results]
            return res


if __name__ == '__main__':
    sents = ["This is a test.", "How are you?", "The weather is good.", "I am fine.", "I am not fine."]
    labels = [1, 2, 3, 4, 5]

    retriever = Ex_Retriver(sents, labels, encode_method='random')

    res = retriever.search_examples("I'm sad.", top_k=3)

    print(res)
