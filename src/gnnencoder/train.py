import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
from torch.utils.data import random_split

from data_utils import ABSAGNNDataset
from models import MultiHeadGAT, contrastive_loss

nhead = 3
token_dim = 768
hidden_dim = 128
output_dim = 512



def evaluate(args, model, val_loader):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_loss = 0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            token_embedding, cl_adj = batch
            token_embedding, cl_adj = token_embedding.float().to(device), cl_adj.float().to(device)

            dims_representations, avg_representation = model(token_embedding)
            loss = contrastive_loss(dims_representations, cl_adj)
            total_loss += loss.item()
    return total_loss

def train(train_loader, val_loader, args):
    model = MultiHeadGAT(nhead, token_dim, hidden_dim, output_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 添加学习率调度器，每 1 个 epochs 减少学习率的10%
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    
    # 用于保存所有步骤的losses
    all_losses = []

    best_loss = float('inf')
    for epoch in range(args.epoch_num):
        total_loss = 0
        for step, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            token_embedding, cl_adj = batch
            token_embedding, cl_adj = token_embedding.float().to(device), cl_adj.float().to(device)

            dims_representations, avg_representation = model(token_embedding)
            loss = contrastive_loss(dims_representations, cl_adj)

            # 检查loss是否为nan或inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'Skipping batch at step {step} due to invalid loss: {loss.item()}')
                continue  # 跳过当前batch
            
            # 保存当前步骤的loss
            all_losses.append(loss.item())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            print(f'Step {step} loss :{loss.item()}')

        # 调用scheduler.step()更新学习率
        print('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
        scheduler.step()

        print('Epoch: {}, loss: {}'.format(epoch, total_loss))

        # 计算验证集上的 loss
        val_loss = evaluate(args, model, val_loader)
        print('Val loss: {}'.format(val_loss))

        if val_loss < best_loss:
            best_loss = val_loss
            # 判断路径是否存在
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            torch.save(model.state_dict(), args.save_path+'best_gnn_model.pt')
            print('Save model in {}'.format(args.save_path+'best_gnn_model.pt\n'))
            
    # 画出loss的下降曲线
    plt.figure(figsize=(10, 6))
    plt.plot(all_losses, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve over Steps')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(args.save_path+'loss.png')

def main(args):
    # 只用训练集作为检索数据集
    task_path = {'ASPE': '/hy-tmp/workspace/SA-LLM/data/inst/ASPE/train_org.json',
                 'MEMD_AS': '/hy-tmp/workspace/SA-LLM/data/inst/MEMD/AS/train_org.json',
                 'MEMD_AOS': '/hy-tmp/workspace/SA-LLM/data/inst/MEMD/AOS/train_org.json',
                 'MEMD_ACOS': '/hy-tmp/workspace/SA-LLM/data/inst/MEMD/ACOS/train_org.json'}
    
    if args.task not in ['ASPE', 'MEMD_AS', 'MEMD_AOS', 'MEMD_ACOS']:
        raise NotImplementedError

    dataset = ABSAGNNDataset(task_path[args.task], args)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    print('Start training...')
    print('Train size: {}, Val size: {}'.format(train_size, val_size))   
    train(train_loader, val_loader, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN model')
    parser.add_argument('--task', type=str, default='ASPE', help='Task name')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epoch_num', type=int, default=10, help='Number of epochs')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    # note 由于 shuffle，这里的 batch_size 必须足够大，才能确保表征模型学到泛化性
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--save_path', type=str, default='./', help='Path to save the model')
    parser.add_argument('--lig_threshold', type=float, default=0.2, help='Threshold for linguistic similarity')
    parser.add_argument('--sent_threshold', type=float, default=0.6, help='Threshold for sentiment similarity')
    # parser.add_argument('--struct_threshold', type=float, default=0.3, help='Threshold for structural similarity')
    args = parser.parse_args()
    
    main(args)