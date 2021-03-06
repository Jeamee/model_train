set -Eeuox
EPOCH=6
DATE=0301
DECODER=crf
BS=4
MAX_LEN=1536
MODEL="funnel-transformer/xlarge"
python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --warmup_ratio 0.01 \
--step_scheduler_metric f1 --trans_lr 1e-5 --other_lr 1e-5 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log \
--ckpt /workspace/xlarge_crf_bs4_ml1536_0301/model_6.bin_epoch0
