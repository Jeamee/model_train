set -Eeuox
python ../src/train.py --fold 3 --model "xlnet-base-cased" --warmup_ratio 0.05 --decoder crf --step_scheduler_metric f1 --trans_lr 5e-5 --other_lr 1e-3 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input /workspace/feedback-prize-2021/ --output ~/autodl-tmp/feedback-prize-2021_base_xlnet_crf_0225 --log ../logs/train_base_xlnet_crf_ep3_0225.log
#python train.py --fold 1 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 2 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 3 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 4 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
