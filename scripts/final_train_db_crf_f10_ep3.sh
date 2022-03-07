set -Eeuox
EPOCH=3
DATE=0307
DECODER=crf
BS=4
MAX_LEN=1024
MODEL="microsoft/deberta-v2-xlarge"

python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --freeze 1 \
--step_scheduler_metric f1 --trans_lr 1e-5 --other_lr 1e-3 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log

BEST=/workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE}/model_${EPOCH}.bin_best
BS=2
MAX_LEN=1536
python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --freeze 0 --freeze_method soft --finetune  --warmup_ratio 0.005 \
--step_scheduler_metric f1 --trans_lr 3e-6 --other_lr 2e-4 --epochs 1 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log \
--ckpt ${BEST}