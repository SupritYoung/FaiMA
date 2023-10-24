

python src/gnnencoder/train.py \
    --task ASPE \
    --save_path ./checkpoints/gnn/memd_as/ \
    --batch_size 128 \
    --max_len 128 \
    --epoch_num 10 \
    --lr 1e-4 \
    --lig_threshold 0.43 \
    --sent_threshold 0.8