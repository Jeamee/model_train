set -Eeuox
EPOCH=6
DATE=0301
DECODER=softmax
BS=16
MAX_LEN=1024
MODEL="uw-madison/yoso-4096"
python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --freeze 10 \
--step_scheduler_metric f1 --trans_lr 1e-4 --other_lr 1e-3 --epochs 40 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH} \
