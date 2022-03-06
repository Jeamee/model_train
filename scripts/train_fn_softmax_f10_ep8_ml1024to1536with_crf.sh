set -Eeuox
EPOCH=8
DATE=0306
DECODER=crf
BS=4
MAX_LEN=1536
MODEL="funnel-transformer/xlarge"
python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER \
--step_scheduler_metric f1 --trans_lr 4e-6 --other_lr 1e-5 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log \
--ckpt /workspace/
