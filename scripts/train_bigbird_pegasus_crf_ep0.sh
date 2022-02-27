set -Eeuox
python ../src/train.py --fold 0 --model "google/bigbird-pegasus-large-arxiv" --log_loss --decoder softmax --step_scheduler_metric f1 --trans_lr 1e-6 --other_lr 1e-3 --epochs 10 --max_len 1536 --batch_size 1 --valid_batch_size 2 --input ~/autodl-tmp/feedback-prize-2021/ --output ~/autodl-tmp/feedback-prize-2021_base_bigbird_pegasus_crf_0225 --log ../logs/train_base_bigbird_pagasus_crf_0225.log
#python train.py --fold 1 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 2 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 3 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 4 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
shutdown
