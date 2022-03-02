set -Eeuox
EPOCH=2
DATE=0301
DECODER=crf
BS=4
MAX_LEN=1024
MODEL="allenai/longformer-large-4096"
python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --finetune --warmup_ratio 0.001 \
--step_scheduler_metric f1 --trans_lr 1e-6 --other_lr 1e-5 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log \
--ckpt /workspace/longformer-large-4096_crf_bs4_ml1536_0301/model_2.bin_epoch4
