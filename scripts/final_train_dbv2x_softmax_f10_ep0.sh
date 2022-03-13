set -Eeuox
EPOCH=0
DATE=0313
DECODER=softmax
BS=16
MAX_LEN=1536
MODEL="microsoft/deberta-v2-xlarge"

python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --freeze 1 --freeze_method soft  --gradient_ckpt --warmup_ratio 0.1 \
--step_scheduler_metric f1 --trans_lr 4e-6 --other_lr 1e-3 --epochs 15 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--seed 12 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log
