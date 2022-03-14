set -Eeuox
EPOCH=3
DATE=0314
DECODER=crf
BS=16
CLASS=easy
MAX_LEN=1536
MODEL="microsoft/deberta-v3-large"

python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --freeze 1 --freeze_method soft  --gradient_ckpt --warmup_ratio 0.075 \
--step_scheduler_metric f1 --trans_lr 1e-5 --other_lr 1e-3 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--seed 12 \
--input /workspace/feedback-prize-2021 \
--input_csv /workspace/feedback-prize-2021/train_fold1.csv \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE}_$CLASS \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log \
