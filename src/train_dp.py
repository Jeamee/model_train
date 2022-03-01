import shutil
from pathlib import Path

transformers_path = Path("/opt/conda/lib/python3.7/site-packages/transformers")

input_dir = Path("../src/deberta-v2-3-fast-tokenizer")

convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path/convert_file.name

if conversion_path.exists():
    conversion_path.unlink()

shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"

for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
    filepath = deberta_v2_path/filename
    if filepath.exists():
        filepath.unlink()

    shutil.copy(input_dir/filename, filepath)


import sys
sys.path.append("/workspace/tez")

import argparse
import os
import random
import warnings
import logging
import time

import numpy as np
import pandas as pd
import tez
import torch
import deepspeed
import torch.nn as nn
from tqdm import tqdm
from math import ceil
from tez import enums
from copy import deepcopy
from sklearn import metrics
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
from torchcrf import CRF
from torch.utils.tensorboard import SummaryWriter


from utils import prepare_training_data, target_id_map, id_target_map, span_target_id_map, span_id_target_map, GradualWarmupScheduler, ReduceLROnPlateau, span_decode
from utils import biaffine_decode, FeedbackDatasetValid, Collate, score_feedback_comp
from dice_loss import DiceLoss
from focal_loss import FocalLoss
from sce import SCELoss

warnings.filterwarnings("ignore")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--trans_lr", type=float, required=True)
    parser.add_argument("--other_lr", type=float, required=True)
    parser.add_argument("--dynamic_merge_layers", action="store_true", required=False)
    parser.add_argument("--merge_layers_num", type=int, default=-2, required=False)
    parser.add_argument("--attack", action="store_true", required=False)
    parser.add_argument("--log_loss", action="store_true", required=False)
    parser.add_argument("--step_scheduler_metric", default="train_f1", type=str, required=False)
    parser.add_argument("--output", type=str, default="../model", required=False)
    parser.add_argument("--input", type=str, default="", required=True)
    parser.add_argument("--ckpt", type=str, default="", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=8, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=8, required=False)
    parser.add_argument("--epochs", type=int, default=20, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--log", type=str, default="train.log", required=False)
    parser.add_argument("--loss", type=str, default="ce", required=False)
    parser.add_argument("--label_smooth", type=float, default=0.0, required=False)
    parser.add_argument("--warmup_ratio", type=float, default=0.05, required=False)
    parser.add_argument("--sce_alpha", type=float, required=False)
    parser.add_argument("--sce_beta", type=float, required=False)
    parser.add_argument("--decoder", type=str, default="softmax", required=False)
    return parser.parse_args()


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FeedbackDataset:
    def __init__(self, samples, max_len, tokenizer, use4span=False, use4biaf=False, use4bb=False):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)
        self.use4span = use4span
        self.use4biaf = use4biaf
        #self.tokenizer.padding_side = "right"
        
        self._init()
    
    def _init(self):
        new_samples = []
        for sample in tqdm(self.samples, total=len(self.samples)):
            input_ids = sample["input_ids"]
            input_labels = sample["input_labels"]
            input_labels = [target_id_map[x] for x in input_labels]
            other_label_id = target_id_map["O"]
            padding_label_id = target_id_map["PAD"]
            
            input_ids = [self.tokenizer.cls_token_id] + input_ids
            input_labels = [other_label_id] + input_labels

            if len(input_ids) > self.max_len - 1:
                input_ids = input_ids[: self.max_len - 1]
                input_labels = input_labels[: self.max_len - 1]

            # add end token id to the input_ids
            input_ids = input_ids + [self.tokenizer.sep_token_id]
            input_labels = input_labels + [other_label_id]

            attention_mask = [1] * len(input_ids)
            
            padding_length = self.max_len - len(input_ids)
            if padding_length > 0:
                #if self.tokenizer.padding_side == "right":
                input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
                input_labels = input_labels + [padding_label_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                #else:
                #    input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
                #    input_labels = [padding_label_id] * padding_length + input_labels
                #    attention_mask = [0] * padding_length + attention_mask
            
            targets = torch.tensor(input_labels, dtype=torch.long)
            if self.use4span:
                length = len(input_labels)
                start_labels, end_labels = [0] * length, [0] * length
                skip_tokens = ["O", "PAD"]
                idx = 0
                while idx < length:
                    label = id_target_map[input_labels[idx]]
                    if label in skip_tokens:
                        idx += 1
                        continue
                    
                    label = label[2:]
                    start_labels[idx] = span_target_id_map[label]
                    next_idx = idx + 1
                    while next_idx < length:
                        next_label = id_target_map[input_labels[next_idx]]
                        if len(next_label) > 3:
                            if next_label[2:] == label:
                                next_idx += 1
                                if next_idx == length:
                                    print(f"add end {span_target_id_map[label]}")
                                    end_labels[next_idx - 1] = span_target_id_map[label]
                                continue
                                
                        end_labels[next_idx - 1] = span_target_id_map[label]
                        break
                    idx = next_idx
                
                targets = [torch.tensor(input_labels, dtype=torch.long), torch.tensor(start_labels, dtype=torch.long).contiguous(), torch.tensor(end_labels, dtype=torch.long)]
            
            
            sample = {
                "ids": torch.tensor(input_ids, dtype=torch.long),
                "mask": torch.tensor(attention_mask, dtype=torch.long),
                "targets": targets,
            }
            
            new_samples.append(sample)
        self.samples = new_samples
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.samples[idx]

class Biaffine(nn.Module):
    def __init__(self, in_size, out_size, bias_x=True, bias_y=True):
        super().__init__()
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.out_size = out_size
        self.U = torch.nn.Parameter(torch.randn(in_size + int(bias_x), out_size ,in_size + int(bias_y)))
        # self.U1 = self.U.view(size=(in_size + int(bias_x),-1))
        #U.shape = [in_size,out_size,in_size]  
    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), dim=-1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), dim=-1)
        
        bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)
        return bilinar_mapping


class FeedbackModel(nn.Module):
    def __init__(self,
        model_name,
        max_len=4096,
        num_labels=15,
        loss="ce",
        decoder="softmax",
        dynamic_merge_layers=False,
        span_num_labels=8,
        sce_alpha=4,
        sce_beta=1,
        label_smooth=0.01,
        ):
        self.num_labels = num_labels
        self.max_len = max_len
        self.merge_layers_num = merge_layers_num
        self.dynamic_merge_layers = dynamic_merge_layers
        self.sce_alpha=sce_alpha,
        self.sce_beta=sce_beta,
        self.label_smooth=label_smooth,


        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
            }
        )
        
        if self.model_name == "microsoft/deberta-v3-large":
            config.update({"max_position_embeddings": 1536})
        
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
            
        if self.dynamic_merge_layers:
            self.layer_logits = nn.Linear(config.hidden_size, 1)
        
        if self.decoder == "span":
            self.start_fc = nn.Linear(config.hidden_size, span_num_labels)
            self.end_fc = nn.Linear(config.hidden_size, span_num_labels)
        else:
            self.output = nn.Linear(config.hidden_size, self.num_labels)
            if self.decoder == "crf":
                self.crf = CRF(num_tags=num_labels, batch_first=True)
        
        
        if loss == "ce":
            self.loss_layer = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        elif loss == "sce":
            self.loss_layer = SCELoss(sce_alpha, sce_beta, num_classes=num_labels if self.decoder != "span" else span_num_labels, label_smooth=label_smooth)
        else:
            raise ValueError("loss set error, must in [ce, sce]")



    def forward(self, ids, mask, token_type_ids=None, targets=None):
        if token_type_ids:
            transformer_out = self.transformer(ids, mask, token_type_ids, output_hidden_states=self.dynamic_merge_layers)
        else:
            transformer_out = self.transformer(ids, mask, output_hidden_states=self.dynamic_merge_layers)
        
        if self.decoder == "crf" and transformer_out.last_hidden_state.shape[1] != ids.shape[1]:
            mask_add = torch.zeros((mask.shape[0],  transformer_out.hidden_states[-1].shape[1] - ids.shape[1])).to(mask.device)
            mask = torch.cat((mask, mask_add), dim=-1)
            
        if self.dynamic_merge_layers:
            layers_output = torch.cat([torch.unsqueeze(layer, 2) for layer in transformer_out.hidden_states[self.merge_layers_num:]], dim=2)
            layers_logits = self.layer_logits(layers_output)
            layers_weights = torch.transpose(torch.softmax(layers_logits, dim=-1), 2, 3)
            sequence_output = torch.squeeze(torch.matmul(layers_weights, layers_output), 2)
        else:
            sequence_output = transformer_out.last_hidden_state
            
        if self.log_loss:
            sequence_output = self.layer_norm(sequence_output)
            
        sequence_output = self.dropout(sequence_output)
        
        if self.decoder == "softmax":
            logits1 = self.output(self.dropout1(sequence_output))
            logits2 = self.output(self.dropout2(sequence_output))
            logits3 = self.output(self.dropout3(sequence_output))
            logits4 = self.output(self.dropout4(sequence_output))
            logits5 = self.output(self.dropout5(sequence_output))
            logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        elif self.decoder == "crf":
            logits = self.output(self.dropout2(sequence_output))
        elif self.decoder == "span":
            sequence_output1 = self.dropout1(sequence_output)
            sequence_output2 = self.dropout2(sequence_output)
            sequence_output3 = self.dropout3(sequence_output)
            sequence_output4 = self.dropout4(sequence_output)
            sequence_output5 = self.dropout5(sequence_output)
            
            start_logits1 = self.start_fc(sequence_output1)
            start_logits2 = self.start_fc(sequence_output2)
            start_logits3 = self.start_fc(sequence_output3)
            start_logits4 = self.start_fc(sequence_output4)
            start_logits5 = self.start_fc(sequence_output5)
            start_logits = (start_logits1 + start_logits2 + start_logits3 + start_logits4 + start_logits5) / 5
            
            end_logits1 = self.end_fc(sequence_output1)
            end_logits2 = self.end_fc(sequence_output2)
            end_logits3 = self.end_fc(sequence_output3)
            end_logits4 = self.end_fc(sequence_output4)
            end_logits5 = self.end_fc(sequence_output5)
            end_logits = (end_logits1 + end_logits2 + end_logits3 + end_logits4 + end_logits5) / 5
            
            logits = (start_logits, end_logits)
        
        probs = None
        if self.decoder == "softmax":
            probs = torch.softmax(logits, dim=-1)
        elif self.decoder == "crf":
            probs = self.crf.decode(emissions=logits, mask=mask.byte())
        elif self.decoder == "span":
            probs = span_decode(start_logits, end_logits)
        else:
            raise ValueException("except decoder in [softmax, crf]")
        loss = 0
        metric = {}

        if targets is not None:
            if self.decoder == "softmax":
                loss1 = self.loss(logits1, targets, attention_mask=mask)
                loss2 = self.loss(logits2, targets, attention_mask=mask)
                loss3 = self.loss(logits3, targets, attention_mask=mask)
                loss4 = self.loss(logits4, targets, attention_mask=mask)
                loss5 = self.loss(logits5, targets, attention_mask=mask)
                loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            elif self.decoder == "crf":
                targets = targets * mask
                loss = -1. * self.crf(emissions=logits, tags=targets, mask=mask.byte(), reduction='mean')
            elif self.decoder == "span":
                targets, start_targets, end_targets = targets
                
                start_loss1 = self.loss(start_logits1, start_targets, attention_mask=mask)
                start_loss2 = self.loss(start_logits2, start_targets, attention_mask=mask)
                start_loss3 = self.loss(start_logits3, start_targets, attention_mask=mask)
                start_loss4 = self.loss(start_logits4, start_targets, attention_mask=mask)
                start_loss5 = self.loss(start_logits5, start_targets, attention_mask=mask)
                start_loss = (start_loss1 + start_loss2 + start_loss3 + start_loss4 + start_loss5) / 5
                
                end_loss1 = self.loss(end_logits1, end_targets, attention_mask=mask)
                end_loss2 = self.loss(end_logits2, end_targets, attention_mask=mask)
                end_loss3 = self.loss(end_logits3, end_targets, attention_mask=mask)
                end_loss4 = self.loss(end_logits4, end_targets, attention_mask=mask)
                end_loss5 = self.loss(end_logits5, end_targets, attention_mask=mask)
                end_loss = (end_loss1 + end_loss2 + end_loss3 + end_loss4 + end_loss5) / 5
                
                loss = start_loss + end_loss
            else:
                raise ValueException("except decoder in [softmax, crf]")
            
            f1 = self.monitor_metrics(probs, targets, attention_mask=mask)["f1"]
            metric["f1"] = f1
        
        if self.log_loss:
            loss = torch.log(loss)
        
        return {
            "preds": probs,
            "logits": logits,
            "loss": loss,
            "metric": metric
        }


    def loss(self, outputs, targets, attention_mask):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        active_loss = attention_mask.view(-1) == 1
        active_logits = outputs.view(-1, self.num_labels if self.decoder not in ["span", "biaffine"] else span_num_labels)
        true_labels = targets.view(-1)
        outputs = active_logits.argmax(dim=-1)
        idxs = np.where(active_loss.cpu().numpy() == 1)[0]
        active_logits = active_logits[idxs]
        true_labels = true_labels[idxs].to(torch.long)

        loss = self.loss_layer(active_logits, true_labels)
        return loss

    def monitor_metrics(self, outputs, targets, attention_mask):
        active_loss = (attention_mask.view(-1) == 1).cpu().numpy()

        idxs = np.where(active_loss == 1)[0]
        
        true_labels = targets.view(-1).cpu().numpy()
        
        
        if self.decoder in ["softmax", "span", "biaffine"]:
            active_logits = outputs.view(-1, self.num_labels)
            outputs = active_logits.argmax(dim=-1).cpu().numpy()[idxs]
        elif self.decoder == "crf":
            outputs = torch.Tensor([output + [0] * (self.max_len - len(output)) for output in outputs])
            outputs = outputs.view(-1).cpu().numpy()[idxs]
        else:
            raise ValueException("except decoder in [softmax, crf]")
            
        f1_score = metrics.f1_score(true_labels[idxs], outputs, average="macro")
        return {"f1": f1_score}


def set_diff_lr(
    model,
    transformer_learning_rate=1e-5,
    other_learning_rate=1e-3
    ):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias"]

    transformer_param_optimizer = []
    crf_param_optimizer = []
    other_param_optimizer = []

    for name, para in param_optimizer:
        space = name.split('.')
        if space[0] == 'transformer':
            transformer_param_optimizer.append((name, para))
        elif space[0] == 'crf':
            crf_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        {"params": [p for n, p in transformer_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01, 'lr': transformer_learning_rate},
        {"params": [p for n, p in transformer_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': transformer_learning_rate},
        # crf模块，差分学习率
        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01, 'lr': 0.01},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': 0.01},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01, 'lr': other_learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': other_learning_rate},
    ]

    return optimizer_grouped_parameters

def process_output(output):
    for key, val in output.items():
        if isinstance(val, torch.Tensor):
            output[key] = val.cpu().detach().numpy()
    return output


def set_log(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    logfile = log_file
    fh = logging.FileHandler(logfile, mode="a")
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    logger.addHandler(fh)
    logger.addHandler(ch)

if __name__ == "__main__":
    NUM_JOBS = 14
    args = parse_args()
    seed_everything(43)
    set_log(args.log)
    tb_writer = SummaryWriter(log_dir=args.tb_log, flush_secs=10)
    os.makedirs(args.output, exist_ok=True)

    df = pd.read_csv(os.path.join(args.input, "train_folds.csv"))

    train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)
    if args.model == "microsoft/deberta-v3-large":
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    training_samples = prepare_training_data(train_df, tokenizer, args, num_jobs=NUM_JOBS)
    valid_samples = prepare_training_data(valid_df, tokenizer, args, num_jobs=NUM_JOBS)

    train_dataset = FeedbackDataset(training_samples, args.max_len, tokenizer, use4span=args.decoder == "span", use4biaf=args.decoder == "biaffine")
    valid_dataset = FeedbackDatasetValid(valid_samples, 4096, tokenizer)
    collate = Collate(tokenizer)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, num_workers=14, collate_fn=collate)
    
    
    num_labels = len(target_id_map) - 1
    span_num_labels = int((len(target_id_map) - 2) // 2) + 1
    model = FeedbackModel(
        max_len=args.max_len,
        model_name=args.model,
        dynamic_merge_layers=args.dynamic_merge_layers,
        merge_layers_num=args.merge_layers_num,
        num_labels=num_labels,
        span_num_labels=span_num_labels,
        loss=args.loss,
        sce_alpha=args.sce_alpha,
        sce_beta=args.sce_beta,
        label_smooth=args.label_smooth,
        decoder=args.decoder,
    )

    best_score = None
    counter = 0

    params = set_diff_lr(model, transformer_learning_rate=args.trans_lr, other_learning_rate=args.other_lr,)
    
    model_engine, _, training_dataloader, _ = deepspeed.initialize(args=args,
        model=model,
        model_parameters=params,
        training_data=train_dataset
        )

    start_epoch = 0
    if args.ckpt:
        _, client_sd = model_engine.load_checkpoint(args.ckpt, args.ckpt_id)
        start_epoch = client_sd["epoch"]

    steps_per_epoch = len(training_dataloader)
    for ep in range(args.epochs - start_epoch):
        ep += start_epoch
        tk0 = tqdm(training_dataloader, total=steps_per_epoch)
        
        losses = AverageMeter()
        monitor = AverageMeter()
        model_engine.train()
        for step, batch in tk0:
            #forward() method

            output = model_engine(batch)

            loss = output["loss"]
            metric = output["metric"]
            losses.update(loss.item(), training_dataloader.batch_size)
            monitor.update(metric, training_dataloader.batch_size)
            tb_writer.add_scalar("cur_f1", metric, step + ep * steps_per_epoch)
            tb_writer.add_scalar("avg_f1", monitor.avg, step + ep * steps_per_epoch)
            #runs backpropagation
            model_engine.backward(loss)
            #weight update
            model_engine.step()
            
            tk0.set_postfix(loss=losses.avg, stage="train", f1=monitor.avg)
        tk0.close()

        client_sd['epoch'] = ep
        ckpt_id = losses.avg
        model_engine.save_checkpoint(args.output, ckpt_id, client_sd = client_sd)

        model_engine.eval()

        tk1 = tqdm(valid_dataloader, total=len(valid_dataloader))
        preds_iter = []
        for _, batch in tk1:
            output = model_engine(batch)
            preds_iter.append(process_output(output))
            tk0.set_postfix(stage="test")
        tk1.close()

        final_preds = []
        final_scores = []
        for output in preds_iter:
            if args.direct_output:
                pred_class = output["preds"]
                pred_scrs = [[1] * len(_) for _ in pred_class]
            else:
                pred_class = np.argmax(output["preds"], axis=2)
                pred_scrs = np.max(output["preds"], axis=2)
            
            for pred, pred_scr in zip(pred_class, pred_scrs):
                final_preds.append(pred if isinstance(pred, list) else pred.tolist())
                final_scores.append(pred_scr if isinstance(pred_scr, list) else pred_scr.tolist())

        for j in range(len(valid_samples)):
            tt = [id_target_map[p] for p in final_preds[j][1:]]
            tt_score = final_scores[j][1:]
            valid_samples[j]["preds"] = tt
            valid_samples[j]["pred_scores"] = tt_score

        submission = []
        min_thresh = {
            "Lead": 9,
            "Position": 5,
            "Evidence": 14,
            "Claim": 3,
            "Concluding Statement": 11,
            "Counterclaim": 6,
            "Rebuttal": 4,
        }
        proba_thresh = {
            "Lead": 0.7,
            "Position": 0.55,
            "Evidence": 0.65,
            "Claim": 0.55,
            "Concluding Statement": 0.7,
            "Counterclaim": 0.5,
            "Rebuttal": 0.55,
        }

        for _, sample in enumerate(valid_samples):
            preds = sample["preds"]
            offset_mapping = sample["offset_mapping"]
            sample_id = sample["id"]
            sample_text = sample["text"]
            sample_pred_scores = sample["pred_scores"]

            # pad preds to same length as offset_mapping
            if len(preds) < len(offset_mapping):
                preds = preds + ["O"] * (len(offset_mapping) - len(preds))
                sample_pred_scores = sample_pred_scores + [0] * (len(offset_mapping) - len(sample_pred_scores))

            idx = 0
            phrase_preds = []
            while idx < len(offset_mapping):
                start, _ = offset_mapping[idx]
                if preds[idx] != "O":
                    label = preds[idx][2:]
                else:
                    label = "O"
                phrase_scores = []
                phrase_scores.append(sample_pred_scores[idx])
                idx += 1
                while idx < len(offset_mapping):
                    if label == "O":
                        matching_label = "O"
                    else:
                        matching_label = f"I-{label}"
                    if preds[idx] == matching_label:
                        _, end = offset_mapping[idx]
                        phrase_scores.append(sample_pred_scores[idx])
                        idx += 1
                    else:
                        break
                if "end" in locals():
                    phrase = sample_text[start:end]
                    phrase_preds.append((phrase, start, end, label, phrase_scores))

            temp_df = []
            for phrase_idx, (phrase, start, end, label, phrase_scores) in enumerate(phrase_preds):
                word_start = len(sample_text[:start].split())
                word_end = word_start + len(sample_text[start:end].split())
                word_end = min(word_end, len(sample_text.split()))
                ps = " ".join([str(x) for x in range(word_start, word_end)])
                if label != "O":
                    if sum(phrase_scores) / len(phrase_scores) >= proba_thresh[label]:
                        temp_df.append((sample_id, label, ps))

            temp_df = pd.DataFrame(temp_df, columns=["id", "class", "predictionstring"])

            submission.append(temp_df)

        submission = pd.concat(submission).reset_index(drop=True)
        submission["len"] = submission.predictionstring.apply(lambda x: len(x.split()))

        def threshold(df):
            df = df.copy()
            for key, value in min_thresh.items():
                index = df.loc[df["class"] == key].query(f"len<{value}").index
                df.drop(index, inplace=True)
            return df

        submission = threshold(submission)

        # drop len
        submission = submission.drop(columns=["len"])

        scr = score_feedback_comp(submission, valid_df, return_class_scores=True)
        logging.info(f"epoch {ep} total:{scr}")

        epoch_score = scr[0]
        
        score = np.copy(epoch_score)

        if best_score is None:
            best_score = score
            best_epoch = ep
        elif score < best_score + 0.0005:
            counter += 1
            logging.info(f"epoch {ep} EarlyStopping counter: {counter} out of {5}")
            if counter >= 5:
                break
        else:
            best_score = score
            best_epoch = epoch
            counter = 0