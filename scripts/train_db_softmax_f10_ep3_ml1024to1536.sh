set -Eeuox
EPOCH=3
DATE=0304
DECODER=softmax
BS=4
MAX_LEN=1536
MODEL="microsoft/deberta-v3-large"
python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --finetune --freeze 1  --lower_freeze 0 \
--step_scheduler_metric f1 --trans_lr 4e-6 --other_lr 4e-6 --epochs 2 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE}-1 \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log \
--ckpt /workspace/model_3.bin_epoch1
