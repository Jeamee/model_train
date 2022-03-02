set -Eeuox
EPOCH=4
DATE=0301
DECODER=crf
BS=2
MAX_LEN=1536
EP=5
MODEL="microsoft/deberta-v3-large"
python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --finetune --warmup_ratio 0.01 --freeze 0 \
--step_scheduler_metric f1 --trans_lr 2e-6 --other_lr 1e-5 --epochs $EP --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log \
--ckpt /workspace/deberta-v3-large_crf_bs4_ml1024_0301/model_${EPOCH}.bin_epochs
