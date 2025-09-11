import os
import argparse
import yaml
import numpy as np
import random
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, recall_score

def compute_gmean(labels_np, preds_np):
    from sklearn.metrics import confusion_matrix
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(labels_np, preds_np).ravel()
    # 计算 Sensitivity (召回率)
    if (tp + fn == 0):
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)
    # 计算 Specificity (特异度)
    if (tn + fp == 0):
        specificity = 0
    else:
        specificity = tn / (tn + fp)
    # 计算 G-Mean
    gmean = np.sqrt(sensitivity * specificity)
    return gmean
    
def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp')
#     parser.add_argument('--gamma', type=float, default=1)
#     parser.add_argument('--C', type=int, default=1)
#     parser.add_argument('--K', type=int, default=1)
    args_input = parser.parse_args()
    config_path = '../config/'+args_input.dataset+'.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    args = argparse.Namespace(**config)
#     args.gamma = args_input.gamma
#     args.C = args_input.C
#     args.K = args_input.K
    print('----------------------------------')
    print('              args')
    print('----------------------------------')
    print(f'dataset:\t{args.dataset}')
    print(f'seed:\t{args.seed}')
    print(f'epoch:\t{args.epoch}')
    print(f'early_stop:\t{args.early_stop}')
    print(f'lr:\t{args.lr}')
    print(f'weigth_decay:{args.weight_decay}')
    print(f'gamma:\t{args.gamma}')
    print(f'C:\t{args.C}')
    print(f'K:\t{args.K}')
    print(f'intra_dim:\t{args.intra_dim}')
    print(f'dropout:\t{args.dropout}')
    print(f'cuda:\t{args.cuda}')
    print('----------------------------------')
    return args

class EarlyStop():
    def __init__(self, early_stop, if_more=True) -> None:
        self.best_eval = 0
        self.best_epoch = 0
        self.if_more = if_more
        self.early_stop = early_stop
        self.stop_steps = 0
    
    def step(self, current_eval, current_epoch):
        do_stop = False
        do_store = False
        if self.if_more:
            if current_eval > self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        else:
            if current_eval < self.best_eval:
                self.best_eval = current_eval
                self.best_epoch = current_epoch
                self.stop_steps = 1
                do_store = True
            else:
                self.stop_steps += 1
                if self.stop_steps >= self.early_stop:
                    do_stop = True
        return do_store, do_stop

def conf_gmean(conf):
	tn, fp, fn, tp = conf.ravel()
	return (tp*tn/((tp+fn)*(tn+fp)))**0.5
def prob2pred(prob, threshhold=0.5):
    pred = np.zeros_like(prob, dtype=np.int32)
    pred[prob >= threshhold] = 1
    pred[prob < threshhold] = 0
    return pred
def evaluate(labels, logits, result_path = ''):
    probs = F.softmax(logits, dim=1)[:,1].cpu().numpy()
    preds = logits.argmax(1).cpu().numpy()
    if len(result_path)>0:
        np.save(result_path+'_result_preds', preds)
        np.save(result_path+'_result_probs', probs)
    conf = confusion_matrix(labels, preds)
    recall = round(recall_score(labels, preds), 4)
    f1_macro = round(f1_score(labels, preds, average='macro'), 4)
    auc = round(roc_auc_score(labels, probs), 4)
    gmean = round(conf_gmean(conf), 4)
    return f1_macro, auc, gmean, recall

def hinge_loss(labels, scores):
    margin = 1
    ls = labels*scores
    
    loss = F.relu(margin-ls)
    loss = loss.mean()
    return loss

def normalize(mx):
	"""
		Row-normalize sparse matrix
		Code from https://github.com/williamleif/graphsage-simple/
	"""
	rowsum = np.array(mx.sum(1)) + 0.01
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx
