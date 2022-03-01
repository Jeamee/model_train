set -Eeuox
EPOCH=0
DATE=0228
DECODER=crf
BS=2
MAX_LEN=1536
MODEL="microsoft/deberta-v3-large"
CUDA_HOME=/usr/local/cuda-11
deepspeed --hostfile=myhostfile ../src/train_dp.py --fold $EPOCH --model $MODEL --decoder $DECODER --step_scheduler_metric f1 --trans_lr 1e-5 --other_lr 1e-3 --epochs 10 --max_len $MAX_LEN --batch_size $BS --valid_batch_size 4 --input /root/feedback-prize-2021 --output /root/autodl-tmp/${MODEL#*/}_${DECODER}_bs${BS}_ml${MAX_LEN}_${DATE} --log ../logs/${MODEL#*/}_${DECODER}_${DATE}_ep${EPOCH}.log --deepspeed --deepspeed_config ../configs/ds_config.json 