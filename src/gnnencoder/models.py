import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, token_dim, hidden_dim, dropout=0.3):
        super(GATLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(token_dim, hidden_dim, bias=True)
        self.attn_fc = nn.Linear(2 * hidden_dim, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_embedding):
        h = self.fc(token_embedding)
        h = self.layer_norm(h)  # Apply layer normalization after the linear transformation
        batch_size, max_len, _ = h.size()

        # Repeat h to have all possible combinations of pairs of nodes
        h_repeat_row = h.unsqueeze(1).repeat(1, max_len, 1, 1).view(batch_size*max_len, max_len, self.hidden_dim)
        h_repeat_col = h.unsqueeze(2).repeat(1, 1, max_len, 1).view(batch_size*max_len, max_len, self.hidden_dim)
        
        concat_h = torch.cat([h_repeat_row, h_repeat_col], dim=-1)
        concat_h = self.dropout(concat_h)  # Apply dropout to the concatenated h

        # Attention mechanism
        attn_weights = self.attn_fc(concat_h).squeeze(-1)
        attn_weights = self.leakyrelu(attn_weights)
        attn_weights = attn_weights.view(batch_size, max_len, max_len)
        attn_weights_normalized = F.softmax(attn_weights, dim=-1)

        h_prime = torch.bmm(attn_weights_normalized, h)
        return h_prime

class MultiHeadGAT(nn.Module):
    def __init__(self, nhead, token_dim, hidden_dim, output_dim=768, dropout=0.3):
        super(MultiHeadGAT, self).__init__()
        
        self.heads = nn.ModuleList()
        for _ in range(nhead):
            self.heads.append(GATLayer(token_dim, hidden_dim, dropout))
        
        self.fc_concat = nn.Linear(nhead * hidden_dim, output_dim)
        self.fcs = nn.ModuleList()
        for _ in range(3):  # 4 dimensions
            self.fcs.append(nn.Linear(output_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, token_embedding):
        out = []
        for attn_head in self.heads:
            out.append(attn_head(token_embedding))
        concat_out = torch.cat(out, dim=2)
        
        # Pooling over the tokens (average pooling)
        sent_embedding = concat_out.mean(dim=1)
        
        # Pass through a dense layer to adjust the dimensions
        sent_embedding = self.fc_concat(sent_embedding)
        sent_embedding = self.layer_norm(sent_embedding)  # Apply layer normalization
        
        dims_out = [fc(self.dropout(sent_embedding)) for fc in self.fcs]  # Apply dropout before passing through the fc
        
        return dims_out, sent_embedding

def contrastive_loss(dims_representations, cl_adj, tau=0.1):
    total_loss = 0
    for dim, reps_dim in enumerate(dims_representations):
        edge_dim_sample = cl_adj[:, :, dim]

        # Calculate similarity matrix
        sim_matrix = torch.einsum('be,ae->bae', reps_dim, reps_dim)  
        # 降温系数，这将使相似性得分在较大范围内波动，有助于模型更好地区分正面和负面对比。
        sim_matrix = F.sigmoid(sim_matrix / tau)

        # Prepare edge_dim_sample to match the shape
        edge_dim_sample_expanded = edge_dim_sample.unsqueeze(-1).expand_as(sim_matrix)

        # Compare with edge_dim_sample_expanded for the loss calculation
        pos_mask = (edge_dim_sample_expanded == 1).float()
        neg_mask = (edge_dim_sample_expanded == 0).float()

        # 为了确保数值稳定性，您可能想为log函数加上一个很小的正值，如1e-8
        pos_loss = -torch.log(sim_matrix + 1e-8) * pos_mask
        pos_loss = torch.masked_select(pos_loss, pos_mask.bool()).mean()

        neg_loss = -torch.log(1.0 - sim_matrix + 1e-8) * neg_mask
        neg_loss = torch.masked_select(neg_loss, neg_mask.bool()).mean()

        total_loss += pos_loss + neg_loss
    return total_loss


if __name__ == "__main__":
    nhead = 4
    token_dim = 768
    hidden_dim = 128
    output_dim = 512

    model = MultiHeadGAT(nhead, token_dim, hidden_dim, output_dim)
    bs = 32
    max_len = 256
    token_embedding_sample = torch.rand(bs, max_len, token_dim)
    edge_embedding_sample = torch.randint(0, 2, (bs, bs, 4)).float()

    dims_representations, avg_representation = model(token_embedding_sample)
    print(avg_representation.shape)
    for dim, reps_dim in enumerate(dims_representations):
        print(reps_dim.shape)
    loss = contrastive_loss(dims_representations, edge_embedding_sample)
    print(loss)
