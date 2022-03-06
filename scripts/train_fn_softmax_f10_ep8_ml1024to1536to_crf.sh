set -Eeuox
EPOCH=8
DATE=0306
DECODER=crf
BS=8
MAX_LEN=1024
MODEL="funnel-transformer/xlarge"
python ../src/train.py --fold $EPOCH --model $MODEL --decoder $DECODER --crf_finetune \
--step_scheduler_metric f1 --trans_lr 3e-7 --other_lr 4e-6 --epochs 1 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 \
--input /workspace/feedback-prize-2021 \
--output /workspace/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} \
--log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log \
--ckpt /workspace/xlarge_softmax_bs4_ml1536_0306/model_8.bin_epoch0
