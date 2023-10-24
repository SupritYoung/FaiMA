# main.py
import argparse
import torch
import time

from Icl.data_utils import data_process
from Icl.evaluation import evaluate
from train_bash import run_exp

train_params_list = [
    "do_train",
    "seed",
    "model_name_or_path",
    "template",
    "lora_target",
    "dataset",
    "dataset_dir",
    "finetuning_type",
    "lora_rank",
    "output_dir",
    "overwrite_output_dir",
    "overwrite_cache",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "lr_scheduler_type",
    "evaluation_strategy",
    "logging_steps",
    "save_steps",
    "save_total_limit",
    "val_size",
    "learning_rate",
    "resume_lora_training",
    "num_train_epochs",
    "load_best_model_at_end",
    "fp16",
    "plot_loss",
    "ddp_find_unused_parameters"
]

def init_args():
    parser = argparse.ArgumentParser(description="Run training and other operations.")
    
    train_args = parser.add_argument_group('train_args', 'Arguments for train_bash.py')
    
    # 添加train_bash.py所需的参数到 train_args 组
    train_args.add_argument("--do_train", action="store_true")
    train_args.add_argument("--seed", type=int, default=42)
    train_args.add_argument("--model_name_or_path", type=str, default="/hy-tmp/llama-2-7b-hf/")
    train_args.add_argument("--template", type=str, default="llama2")
    train_args.add_argument("--lora_target", type=str, default="q_proj,v_proj")
    train_args.add_argument("--dataset", type=str)
    train_args.add_argument("--dataset_dir", type=str, default="./data")
    train_args.add_argument("--finetuning_type", type=str, default="lora")
    train_args.add_argument("--lora_rank", type=int, default=128)
    train_args.add_argument("--output_dir", type=str)
    train_args.add_argument("--overwrite_output_dir", action="store_true")
    train_args.add_argument("--overwrite_cache", action="store_true")
    train_args.add_argument("--per_device_train_batch_size", type=int, default=16)
    train_args.add_argument("--per_device_eval_batch_size", type=int, default=16)
    train_args.add_argument("--gradient_accumulation_steps", type=int, default=4)
    train_args.add_argument("--lr_scheduler_type", type=str, default="cosine")
    train_args.add_argument("--evaluation_strategy", type=str, default="steps")
    train_args.add_argument("--logging_steps", type=int, default=10)
    train_args.add_argument("--save_steps", type=int, default=500)
    train_args.add_argument("--save_total_limit", type=int, default=5)
    train_args.add_argument("--val_size", type=float, default=0.1)
    train_args.add_argument("--learning_rate", type=float, default=8e-5)
    train_args.add_argument("--resume_lora_training", action="store_true")
    train_args.add_argument("--num_train_epochs", type=float, default=5.0)
    train_args.add_argument("--load_best_model_at_end", action="store_true")
    train_args.add_argument("--fp16", action="store_true")
    train_args.add_argument("--plot_loss", action="store_true")
    train_args.add_argument("--ddp_find_unused_parameters", action="store_true")

    data_args = parser.add_argument_group('main_args', 'Arguments for main.py')
    # 添加main.py所需的参数到 data_args 组
    data_args.add_argument("--task", type=str, default="MEMD_AOS")
    # data_args.add_argument("--is_ICL", action="store_true")
    data_args.add_argument("--method", type=str, default="gnn")
    data_args.add_argument("--top_k", type=int, default=5)
    data_args.add_argument("--bert_path", type=str, default="/hy-tmp/bert-base-uncased")
    data_args.add_argument("--sbert_path", type=str, default="/hy-tmp/all-MiniLM-L6-v2/")
    data_args.add_argument("--gnn_path", type=str)

    args = parser.parse_args()
    # 使用字典推导式从 args 中提取 train_args 和 data_args 的参数
    train_params = {param: getattr(args, param) for param in args.__dict__ if param in train_params_list}
    data_params = {param: getattr(args, param) for param in args.__dict__ if param not in train_params_list}

    # 手动添加参数
    if 'dataset' not in train_params.keys() or train_params["dataset"] is None:
        train_params["dataset"] = f'{data_params["task"].lower()}_{data_params["method"].lower()}_train'
    if 'output_dir' not in train_params.keys() or train_params["output_dir"] is None:
        train_params["output_dir"] = f'./checkpoints/{data_params["task"]}_{data_params["method"]}_{time.strftime("%m-%d-%H", time.localtime())}'
    if 'gnn_path' not in data_params.keys() or data_params["gnn_path"] is None:
        data_params["gnn_path"] = f'/hy-tmp/workspace/SA-LLM/checkpoints/gnn/{data_params["task"].lower()}/best_gnn_model.pt'
    data_params["is_ICL"] = False if data_params["method"] == "noicl" else True

    return train_params, data_params
    
def main():
    train_params, data_params = init_args()
    # 加载测试数据
    paths = {
        "bert_path": data_params["bert_path"],
        "sbert_path": data_params["sbert_path"],
        "gnn_path": data_params["gnn_path"]
    }
    test_data = data_process(task=data_params['task'], method=data_params['method'], ICL=data_params['is_ICL'], paths=paths)

    print("\n========================================")
    print("Start training...\n")
    print("========================================\n")
    
    # 开始训练模型
    run_exp(train_params)

    torch.cuda.empty_cache()

    print("\n========================================")
    print("Start evaluating...\n")
    print("========================================\n")

    evaluate(test_data, task=data_params['task'], checkpoint_dir=train_params["output_dir"],
             temperature=0.1, top_p=0.9, finetuning_type=train_params["finetuning_type"])
    
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()