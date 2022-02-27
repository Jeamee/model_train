pip install transformers psutil

set -Eeuox
python ../src/train.py --fold 0 --model "allenai/longformer-large-4096" --label_smooth 0.05 --dynamic_merge_layers --loss sce --sce_alpha 4.0 --sce_beta 1.0 --step_scheduler_metric f1 --trans_lr 1e-5 --other_lr 1e-3 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ~/autodl-tmp/feedback-prize-2021/ --output ~/autodl-tmp/feedback-prize-2021_sce_a4.0b1.0_label_smooth_0212 --log ../logs/train_lm_sce_a4.0b1.0_label_smooth_0212.log
#python train.py --fold 1 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 2 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 3 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 4 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
