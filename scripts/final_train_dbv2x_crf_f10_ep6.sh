set -Eeuox
EPOCH=6
DATE=0312
DECODER=crf
BS=8
MAX_LEN=1536
MODEL="microsoft/deberta-v2-xlarge"

python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --freeze 0 --freeze_method soft  --gradient_ckpt --warmup_ratio 0.01 \
--step_scheduler_metric f1 --trans_lr 4e-6 --other_lr 1e-3 --epochs 14 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--seed 12 \
--accumulation_steps 2 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE}_1 \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log \
--ckpt /workspace/deberta-v2-xlarge_crf_bs16_ml1536_0312/model_6.bin_best
