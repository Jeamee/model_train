set -Eeuox
EPOCH=3
DATE=0304
DECODER=crf
BS=4
MAX_LEN=1536
MODEL="microsoft/deberta-v3-large"
python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --crf_finetune --warmup_ratio 0.01 \
--step_scheduler_metric f1 --trans_lr 2e-6 --other_lr 2e-6 --epochs 1 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log \
--ckpt /workspace/deberta-v3-large_softmax_bs4_ml1536_0304-0/model_3.bin_epoch0
