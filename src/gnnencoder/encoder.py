import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
import re
import json
import torch.nn as nn

from gnnencoder.models import MultiHeadGAT

nhead = 3
token_dim = 768
hidden_dim = 128
output_dim = 512

def parse_output(s):
    groups = re.findall(r'\[(.*?)\]', s)
    elements = [re.split(r',\s*', group) for group in groups]
    return elements

class SentenceEncoder:
    def __init__(self, gnn_model_path, bert_model_path, max_len=256):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 初始化GNN模型并加载权重
        self.gnn_model = MultiHeadGAT(nhead, token_dim, hidden_dim, output_dim).to(self.device)
        if torch.cuda.device_count() > 1:
            self.gnn_model = nn.DataParallel(self.gnn_model, device_ids=[i for i in range(torch.cuda.device_count())])
        self.gnn_model.load_state_dict(torch.load(gnn_model_path))
        self.gnn_model.eval()
        
        # 初始化BERT模型用于编码句子
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        self.bert_model = BertModel.from_pretrained(bert_model_path).to(self.device)
        if torch.cuda.device_count() > 1:
            self.bert_model = nn.DataParallel(self.bert_model)
        self.bert_model.eval()
        
        self.max_len = max_len

    def encode(self, sentences):
        # 1. 使用BERT模型编码句子
        encoded_sentences = self._batch_encode_nodes(sentences)
        # 2. 使用GNN模型进一步编码得到句子表征
        with torch.no_grad():
            dims_representations, avg_representation = self.gnn_model(encoded_sentences)
        return dims_representations, avg_representation

    def _batch_encode_nodes(self, inputs):
        """
        批量编码文本为节点特征
        """
        encoded_inputs = self.tokenizer.batch_encode_plus(inputs, return_tensors='pt', padding='max_length',
                                                          max_length=self.max_len, truncation=True)
        input_ids = encoded_inputs['input_ids'].to(self.device)
        attention_mask = encoded_inputs['attention_mask'].to(self.device)  # 获取attention_mask

        with torch.no_grad():
            embeddings = self.bert_model(input_ids, attention_mask=attention_mask).last_hidden_state  # 使用attention_mask
        return embeddings

if __name__ == '__main__':
    # 使用示例：
    gnn_model_path = 'SA-LLM/checkpoints/gnn/best_model.pt'
    bert_model_path = '/hy-tmp/bert-base-uncased'
    encoder = SentenceEncoder(gnn_model_path, bert_model_path)

    # 假设你有一个句子列表
    sentences_list = ["Moodys gives fifth reason for markets to cheer ; record highs seen", "GVK Power can go up to 14-15 levels : Vijay Bhambwani"]
    # 得到句子的表征
    dims_representations, representations = encoder.encode(sentences_list)
    print(representations.shape)
    # print(representations[0][:100])