import torch
from gnn_models import GNN_CL

bs = 64
F = 768
K = 4
N = 128       # padding 的 max token 值


# Notice
# token_embedding: bs * N * F
# edge_embedding: bs * N * N * F
# adj: bs * N * N   => 自己构造token内部的adj
# mask: bs * N  => 自己padding+构造mask
# Indice: 指示矩阵 K * bs * bs=> 自己构建

model = GNN_CL(input_feature_size=F, edge_size=K, output_size=F, model_name='GAT')

for i in range(10):     # epoch
    for batch in range(100):    # batch
        token_embedding = torch.rand(bs, N, F)
        edge_embedding = torch.rand(bs, N, N, K)
        adj = torch.rand(bs, N, N)
        adj = (adj > 0.5) * torch.ones_like(adj) + (adj <= 0.5) * torch.zeros_like(adj)
        mask = torch.rand(bs, N)
        mask = mask > 0.1
        indices = torch.rand(K, bs, bs)
        indices = indices > 0.8

        out_embedding = model(X=token_embedding, E=edge_embedding, A=adj, M=mask, I=indices)
        print(out_embedding.shape)
        # out_embedding: bs * K * F

        cl_loss = model.cl_loss     # CL loss
        print(cl_loss)