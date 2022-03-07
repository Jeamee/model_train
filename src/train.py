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

import gc
gc.enable()

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
import torch.nn as nn
import bitsandbytes as bnb
from tqdm import tqdm
from math import ceil
from tez import enums
from copy import deepcopy
from sklearn import metrics
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast
#from torchcrf import CRF
from pytorchcrf import CRF


from utils import EarlyStopping, prepare_training_data, target_id_map, id_target_map, span_target_id_map, span_id_target_map, GradualWarmupScheduler, ReduceLROnPlateau, span_decode
from utils import biaffine_decode, Freeze
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
    parser.add_argument("--finetune", action="store_true", required=False)
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
    parser.add_argument("--freeze", type=int, default=10, required=False)
    parser.add_argument("--freeze_method", type=str, default="hard", required=False)
    parser.add_argument("--crf_finetune", action="store_true", required=False)
    parser.add_argument("--lower_freeze", type=float, default=0., required=False)
    parser.add_argument("--finetune_to_1536", action="store_true", required=False)
    
    return parser.parse_args()



class FeedbackDataset:
    def __init__(self, samples, max_len, tokenizer, use4span=False, use4biaf=False, use4bb=False, use4lf=False):
        self.samples = samples
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.length = len(samples)
        self.use4span = use4span
        self.use4biaf = use4biaf
        self.use4lf = use4lf
        #self.tokenizer.padding_side = "right"
        
        self._init()
    
    def _init(self):
        new_samples = []
        for sample in tqdm(self.samples, total=len(self.samples)):
            sample_id = sample["id"]
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
            
            if self.use4lf:
                pad_length = ceil(len(attention_mask) / 512) * 512 - len(attention_mask)
                attention_mask.extend([0] * pad_length)
            
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
    
    
class FeedbackModel(tez.Model):
    def __init__(
        self,
        model_name,
        num_train_steps,
        transformer_learning_rate,
        other_learning_rate,
        num_labels,
        span_num_labels,
        steps_per_epoch,
        dynamic_merge_layers,
        step_scheduler_metric,
        loss="ce",
        sce_alpha=1.0,
        sce_beta=1.0,
        label_smooth=0.0,
        decoder="softmax",
        max_len=4096,
        merge_layers_num=-2,
        log_loss=False,
        warmup_ratio=0.05,
        finetune=False,
        lower_freeze=0.,
        crf_finetune=False,
        mid_linear_dims=128
    ):
        super().__init__()
        self.cur_step = 0
        self.max_len = max_len
        self.transformer_learning_rate = transformer_learning_rate
        self.other_learning_rate = other_learning_rate
        self.dynamic_merge_layers = dynamic_merge_layers
        self.merge_layers_num = merge_layers_num
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.num_labels = num_labels
        self.span_num_labels = span_num_labels
        self.label_smooth = label_smooth
        self.steps_per_epoch = steps_per_epoch
        self.step_scheduler_metric = step_scheduler_metric
        self.decoder = decoder
        self.log_loss=log_loss
        self.warmup_ratio = warmup_ratio
        self.finetune = finetune
        self.lower_freeze = lower_freeze
        self.step_scheduler_after = "batch"
        self.crf_finetune = crf_finetune

        hidden_dropout_prob: float = 0.1
        layer_norm_eps: float = 1e-7

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        config.update(
            {
                "output_hidden_states": True,
                "hidden_dropout_prob": hidden_dropout_prob,
                "layer_norm_eps": layer_norm_eps,
                "add_pooling_layer": False,
                "num_labels": self.num_labels,
                
            }
        )
        
        if self.model_name in ["microsoft/deberta-v3-large", "microsoft/deberta-v2-xlarge", "uw-madison/yoso-4096", "funnel-transformer/xlarge"]:
            logging.info("set max_position_embeddings to 4096")
            config.update({"max_position_embeddings": 4096})
        
        self.transformer = AutoModel.from_pretrained(model_name, config=config)
        
        self.mid_linear = nn.Sequential(
            nn.Linear(config.hidden_size, mid_linear_dims),
            nn.SELU(),
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        if self.lower_freeze > 0:
            params = list(self.transformer.parameters())
            layer_num = int(len(params) * self.lower_freeze)
            for param in params[: layer_num + 1]:
                param.requires_grad = False
            torch.cuda.empty_cache()
            gc.collect()
        
        if self.log_loss:
            self.layer_norm = nn.LayerNorm(config.hidden_size)
            
        if self.dynamic_merge_layers:
            self.layer_logits = nn.Linear(config.hidden_size, 1)
        
        if self.decoder == "span":
            self.start_fc = nn.Linear(config.hidden_size, span_num_labels)
            self.end_fc = nn.Linear(config.hidden_size, span_num_labels)
        else:
            self.output = nn.Linear(mid_linear_dims, self.num_labels)
            if self.decoder == "crf":
                self.crf = CRF(num_tags=num_labels, batch_first=True)
        
        
        if loss == "ce":
            self.loss_layer = nn.CrossEntropyLoss(label_smoothing=label_smooth)
        elif loss == "sce":
            self.loss_layer = SCELoss(sce_alpha, sce_beta, num_classes=num_labels if self.decoder != "span" else span_num_labels, label_smooth=label_smooth)
        else:
            raise ValueError("loss set error, must in [ce, sce]")
            
        if crf_finetune:
            for name, para in self.named_parameters():
                space = name.split('.')
                if space[0] != 'crf':
                    para.requires_grad = False
            
    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]

        lonformer_param_optimizer = []
        crf_param_optimizer = []
        other_param_optimizer = []

        for name, para in param_optimizer:
            space = name.split('.')
            if space[0] == 'transformer':
                lonformer_param_optimizer.append((name, para))
            elif space[0] == 'crf':
                crf_param_optimizer.append((name, para))
            else:
                other_param_optimizer.append((name, para))
                
        crf_lf = 1e-5 if self.finetune else 1e-2
        
        self.optimizer_grouped_parameters = [
            {"params": [p for n, p in lonformer_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01, 'lr': self.transformer_learning_rate},
            {"params": [p for n, p in lonformer_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0, 'lr': self.transformer_learning_rate},
            
            {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01, 'lr': crf_lf},
            {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0, 'lr': crf_lf},

            # 其他模块，差分学习率
            {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01, 'lr': self.other_learning_rate},
            {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0, 'lr': self.other_learning_rate},
        ]
        
        opt = bnb.optim.AdamW8bit(self.optimizer_grouped_parameters, lr=self.transformer_learning_rate)
    
        for module in self.modules():
            if isinstance(module, torch.nn.Embedding):
                bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                    module, 'weight', {'optim_bits': 32}
                )            
        #opt = AdamW(self.optimizer_grouped_parameters, lr=self.transformer_learning_rate)
        return opt
    
    def fetch_scheduler(self):
        if not self.finetune:
            min_lr = [1e-5, 1e-5, 1e-8, 1e-8, 1e-7, 1e-7]
            patience = 10

            sch = GradualWarmupScheduler(
                self.optimizer,
                multiplier=1.1,
                warmup_epoch=int(self.warmup_ratio * self.num_train_steps) ,
                total_epoch=self.num_train_steps)
            logging.info("finetune did not set sch")
            return sch
        
        if self.crf_finetune:
            sch = StepLR(self.optimizer, self.num_train_steps // 4, gamma=0.2)
            logging.info("crf finetune set steplr sch")
        

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
            #outputs = outputs.view(-1).cpu().numpy()[idxs]
            outputs = outputs.reshape(-1).cpu().numpy()[idxs]
        else:
            raise ValueException("except decoder in [softmax, crf]")
            
        f1_score = metrics.f1_score(true_labels[idxs], outputs, average="macro")
        return {"f1": f1_score}

    def forward(self, ids, mask, token_type_ids=None, targets=None, id=None):
        if token_type_ids:
            transformer_out = self.transformer(ids, mask, token_type_ids, output_hidden_states=self.dynamic_merge_layers)
        else:
            transformer_out = self.transformer(ids, mask, output_hidden_states=self.dynamic_merge_layers)
            
        if self.dynamic_merge_layers:
            layers_output = torch.cat([torch.unsqueeze(layer, 2) for layer in transformer_out.hidden_states[self.merge_layers_num:]], dim=2)
            layers_logits = self.layer_logits(layers_output)
            layers_weights = torch.transpose(torch.softmax(layers_logits, dim=-1), 2, 3)
            sequence_output = torch.squeeze(torch.matmul(layers_weights, layers_output), 2)
        else:
            sequence_output = transformer_out.last_hidden_state
            
        if self.log_loss:
            sequence_output = self.layer_norm(sequence_output)
            
        # sequence_output = self.dropout(sequence_output)
        sequence_output = self.mid_linear(sequence_output)
        
        if self.decoder == "softmax":
            logits1 = self.output(self.dropout1(sequence_output))
            logits2 = self.output(self.dropout2(sequence_output))
            logits3 = self.output(self.dropout3(sequence_output))
            logits4 = self.output(self.dropout4(sequence_output))
            logits5 = self.output(self.dropout5(sequence_output))
            logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
        elif self.decoder == "crf":
            sequence_output1 = self.dropout1(sequence_output)
            sequence_output2 = self.dropout2(sequence_output)
            sequence_output3 = self.dropout3(sequence_output)
            sequence_output4 = self.dropout4(sequence_output)
            sequence_output5 = self.dropout5(sequence_output)
            sequence_output = (sequence_output1 + sequence_output2 + sequence_output3 +sequence_output4 + sequence_output5) / 5
            logits = self.output(sequence_output)
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
                if self.crf_finetune:
                    self.train()
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
            if id:
                id = " ".join(id)
                logging.info(f"step: {self.current_train_step}, ids: {id}, f1: {f1}")
        
        if self.log_loss:
            loss = torch.log(loss)
        
        return {
            "preds": probs,
            "logits": logits,
            "loss": loss,
            "metric": metric
        }
    

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
    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(os.path.join(args.input, "train_folds10.csv"))

    train_df = df[df["kfold"] != args.fold].reset_index(drop=True)
    valid_df = df[df["kfold"] == args.fold].reset_index(drop=True)
    if args.model in ["microsoft/deberta-v3-large", "microsoft/deberta-v2-xlarge"]:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.model != "allenai/longformer-large-4096":
        tokenizer.add_tokens("\n", special_tokens=True)
        logging.info("add return token to vocab")
        
    training_samples = prepare_training_data(train_df, tokenizer, args, num_jobs=NUM_JOBS, only_bigger_than_1024=args.finetune_to_1536)
    valid_samples = prepare_training_data(valid_df, tokenizer, args, num_jobs=NUM_JOBS)

    train_dataset = FeedbackDataset(
        training_samples,
        args.max_len,
        tokenizer,
        use4span=args.decoder == "span",
        use4biaf=args.decoder == "biaffine",
        use4lf=args.model == "allenai/longformer-large-4096"
    )

    num_train_steps = int(len(train_dataset) / args.batch_size / args.accumulation_steps * args.epochs)
    
    num_labels = len(target_id_map) - 1
    span_num_labels = int((len(target_id_map) - 2) // 2) + 1
    model = FeedbackModel(
        max_len=args.max_len,
        model_name=args.model,
        num_train_steps=num_train_steps,
        transformer_learning_rate=args.trans_lr,
        other_learning_rate=args.other_lr,
        dynamic_merge_layers=args.dynamic_merge_layers,
        merge_layers_num=args.merge_layers_num,
        step_scheduler_metric=args.step_scheduler_metric,
        num_labels=num_labels,
        span_num_labels=span_num_labels,
        steps_per_epoch=len(train_dataset) / args.batch_size,
        loss=args.loss,
        sce_alpha=args.sce_alpha,
        sce_beta=args.sce_beta,
        label_smooth=args.label_smooth,
        decoder=args.decoder,
        log_loss=args.log_loss,
        warmup_ratio=args.warmup_ratio,
        finetune=args.finetune,
        lower_freeze=args.lower_freeze,
        crf_finetune=args.crf_finetune
    )
    
    if args.model != "allenai/longformer-large-4096":
        model.transformer.resize_token_embeddings(len(tokenizer))
        logging.info("model emb matrix resized")
    
    if args.ckpt:
        model.load(args.ckpt, weights_only=True, strict=False)
        logging.info(f"{args.ckpt}")
        
    
    
    freeze = Freeze(epochs=args.freeze if not args.crf_finetune else 9999, method=args.freeze_method)
    tb_logger = tez.callbacks.TensorBoardLogger(log_dir=f"{args.output}/tb_logs/")
    es = EarlyStopping(
        model_path=os.path.join(args.output, f"model_{args.fold}.bin"),
        valid_df=valid_df,
        valid_samples=valid_samples,
        batch_size=args.valid_batch_size,
        patience=5,
        mode="max",
        delta=0.001,
        save_weights_only=True,
        tokenizer=tokenizer,
        direct_output=args.decoder == "crf",
        bigbird_padding=args.model in ["google/bigbird-roberta-large", "patrickvonplaten/bigbird-roberta-large"]
    )

    model.fit(
        train_dataset,
        train_bs=args.batch_size,
        device="cuda",
        epochs=args.epochs,
        callbacks=[tb_logger, es, freeze],
        fp16=args.model != "uw-madison/yoso-4096",
        attack=args.attack,
        accumulation_steps=args.accumulation_steps,
    )

