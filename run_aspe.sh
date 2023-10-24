

python ./src/main.py \
    --task ASPE \
    --method noicl \
    --do_train \
    --seed 42 \
    --model_name_or_path /hy-tmp/llama-2-7b-hf/ \
    --finetuning_type lora \
    --lora_rank 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --logging_steps 20 \
    --save_steps 100 \
    --val_size 0.1 \
    --learning_rate 8e-5 \
    --num_train_epochs 5.0 \
    --fp16

python ./src/main.py \
    --task ASPE \
    --method random \
    --k 5\
    --do_train \
    --seed 42 \
    --model_name_or_path /hy-tmp/llama-2-7b-hf/ \
    --finetuning_type lora \
    --lora_rank 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --logging_steps 20 \
    --save_steps 100 \
    --val_size 0.1 \
    --learning_rate 8e-5 \
    --num_train_epochs 5.0 \
    --fp16

python ./src/main.py \
    --task ASPE \
    --method sbert \
    --k 5\
    --do_train \
    --seed 42 \
    --model_name_or_path /hy-tmp/llama-2-7b-hf/ \
    --finetuning_type lora \
    --lora_rank 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --logging_steps 20 \
    --save_steps 100 \
    --val_size 0.1 \
    --learning_rate 8e-5 \
    --num_train_epochs 5.0 \
    --fp16

python ./src/main.py \
    --task ASPE \
    --method gnn \
    --k 5\
    --do_train \
    --seed 42 \
    --model_name_or_path /hy-tmp/llama-2-7b-hf/ \
    --finetuning_type lora \
    --lora_rank 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --logging_steps 20 \
    --save_steps 100 \
    --val_size 0.1 \
    --learning_rate 8e-5 \
    --num_train_epochs 5.0 \
    --fp16