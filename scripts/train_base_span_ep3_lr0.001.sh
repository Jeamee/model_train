set -Eeuox
python ../src/train.py --fold 3 --model "allenai/longformer-large-4096" --decoder span --step_scheduler_metric f1 --trans_lr 1e-5 --other_lr 1e-3 --epochs 10 --max_len 1536  --batch_size 4 --valid_batch_size 8 --input ~/autodl-tmp/feedback-prize-2021/ --output ~/autodl-tmp/feedback-prize-2021_base_span_lr1e-3_0217 --log ../logs/train_base_span_lr1e-3_0217_ep3.log 
#python train.py --fold 1 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 2 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 3 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
#python train.py --fold 4 --model ../../data/longformer-large-4096 --trans_lr 1e-5 --other_lr 3e-5 --epochs 10 --max_len 1536 --batch_size 4 --valid_batch_size 4 --input ../../data/feedback-prize-2021/ --output ../feedback-prize-2021-0210
shutdown