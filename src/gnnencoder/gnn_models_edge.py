import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, token_dim, edge_dim, hidden_dim):  # Add hidden_dim as an argument
        super(GATLayer, self).__init__()
        
        self.hidden_dim = hidden_dim  # Store hidden_dim as a class attribute
        
        self.fc = nn.Linear(token_dim + edge_dim, hidden_dim, bias=False)
        self.attn_fc = nn.Linear(2 * hidden_dim, 1)
        self.leakyrelu = nn.LeakyReLU()


    def forward(self, token_embedding, edge_embedding):
        # Take average edge embedding across the batch dimension to get an embedding per token
        avg_edge_embedding = edge_embedding.mean(dim=1, keepdim=True)
        combined_embedding = torch.cat([token_embedding, avg_edge_embedding.repeat(1, token_embedding.size(1), 1)], dim=-1)

        h = self.fc(combined_embedding)
        batch_size = h.size(0)

        # Repeat h to have all possible combinations of pairs of nodes
        max_len = token_embedding.size(1)
        h_repeat_row = h.unsqueeze(1).repeat(1, max_len, 1, 1).view(batch_size*max_len, max_len, self.hidden_dim)
        h_repeat_col = h.unsqueeze(2).repeat(1, 1, max_len, 1).view(batch_size*max_len, max_len, self.hidden_dim)
        
        concat_h = torch.cat([h_repeat_row, h_repeat_col], dim=-1)

        # Attention mechanism
        attn_weights = self.attn_fc(concat_h).squeeze(-1)
        attn_weights = attn_weights.view(batch_size, max_len, max_len)
        attn_weights_normalized = F.softmax(attn_weights, dim=-1)

        # Ensure h is of shape [batch_size, max_len, hidden_dim]
        h = h.view(batch_size, max_len, self.hidden_dim)

        # Weighted combination of h based on attention weights
        h_prime = torch.bmm(attn_weights_normalized, h)

        return h_prime


class MultiHeadGAT(nn.Module):
    def __init__(self, nhead, token_dim, edge_dim, hidden_dim, output_dim):
        super(MultiHeadGAT, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(nhead):
            self.heads.append(GATLayer(token_dim, edge_dim, hidden_dim))  # 添加了out_dim参数
        self.fc = nn.Linear(nhead * hidden_dim, output_dim)

    def forward(self, token_embedding, edge_embedding):
        out = []
        for attn_head in self.heads:
            out.append(attn_head(token_embedding, edge_embedding))
        concat_out = torch.cat(out, dim=2)
        return self.fc(concat_out)


def contrastive_loss(representations, edge_embedding):
    batch_size, _, edge_dim = edge_embedding.shape
    total_loss = 0
    
    for dim in range(edge_dim):
        # 取出该维度下的表示和相似度
        reps_dim = representations[:, dim, :]
        edge_dim_sample = edge_embedding[:, :, dim]
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(reps_dim, reps_dim.transpose(1, 0))
        sim_matrix = F.sigmoid(sim_matrix)

        # 创建相似度标签矩阵
        pos_mask = (edge_dim_sample == 1).float()
        neg_mask = (edge_dim_sample == 0).float()

        # 计算正面对的损失
        pos_loss = -torch.log(sim_matrix) * pos_mask
        pos_loss = torch.masked_select(pos_loss, pos_mask.bool()).mean()

        # 计算负面对的损失
        neg_loss = -torch.log(1.0 - sim_matrix) * neg_mask
        neg_loss = torch.masked_select(neg_loss, neg_mask.bool()).mean()

        # 更新总损失
        total_loss += pos_loss + neg_loss

    return total_loss


if __name__ == "__main__":
    # Define parameters
    nhead = 4
    token_dim = 768
    edge_dim = 4
    hidden_dim = 128
    output_dim = 256

    model = MultiHeadGAT(nhead, token_dim, edge_dim, hidden_dim, output_dim)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Example forward pass
    bs = 32
    max_len = 256
    token_embedding_sample = torch.rand(bs, max_len, token_dim)
    edge_embedding_sample = torch.randint(0, 2, (bs, bs, edge_dim)).float()

    representations = model(token_embedding_sample, edge_embedding_sample)
    loss = contrastive_loss(representations, edge_embedding_sample)

    print(loss)
    # loss.backward()
    # optimizer.step()

