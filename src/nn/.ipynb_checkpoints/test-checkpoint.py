import os
import gc
import time
import numpy as np
from tqdm import tqdm, trange
import datatable as dt
from ..common_path import *
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .utils import (
    seed_everything, roc_score, load_embedding_matrix, EMA, ProcessedData,
    batch2cuda, WXDataset, Collator, read_data,
    undersampling_neg_sample_all_actions
)

import torch
import torch.nn as nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader

from .multideepfm4wx import MultiDeepFM
from .args import get_args

ACTIONS = ["read_comment", "like", "click_avatar", "forward", 'favorite', 'comment', 'follow']

def prepare_data(args):
    processed_data = ProcessedData(args)
    collate_fn = Collator(args)
    return collate_fn, processed_data


def args_setup():
    args = get_args()
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        args.n_gpus = torch.cuda.device_count()
        args.bs *= args.n_gpus
    args.multi_modal_emb_matrix = load_embedding_matrix(
        filepath=args.multi_modal_emb_matrix_path,
        max_vocab_size=args.vocab_size
    )
    args.multi_modal_emb_char_matrix = load_embedding_matrix(
        filepath=args.multi_modal_emb_char_matrix_path,
        max_vocab_size=args.char_vocab_size
    )
    return args



# def predict(config, model, test_dataloader):
#     test_iterator = tqdm(test_dataloader, desc='Predicting', total=len(test_dataloader))
#     test_preds = []

#     test_df = pd.read_csv(config.submit_test_path)
#     if config.debug_data:
#         test_df = test_df.head(1000)

#     model.eval()
#     with torch.no_grad():
#         t = time.time()
#         for batch in test_iterator:
#             batch_cuda, _ = batch2cuda(batch, config)
#             logits = model(**batch_cuda)[4]
#             probs = torch.sigmoid(logits)
#             test_preds.append(probs.detach().cpu())

#     ts = (time.time() - t) * 1000.0 / len(test_df) * 2000.0 / 4
#     print(f'\n>>> Single action average cost time {ts:.4} ms on 2000 samples ...')

#     test_preds = torch.cat(test_preds).numpy()
#     test_df[['read_comment', 'like', 'click_avatar', 'forward']] = test_preds
#     test_df.drop('device', axis=1, inplace=True)
#     submission_path = os.path.join(SUBMISSION_PATH, "nn{}.csv".format(config.seed))
#     test_df.to_csv(submission_path, index=False, encoding='utf8', sep=',')

def predict(config, model, test_dataloader, test_df):
    test_iterator = tqdm(test_dataloader, desc='Predicting', total=len(test_dataloader))
    test_preds = []    

    model.eval()
    with torch.no_grad():
        t = time.time()
        for batch in test_iterator:
            batch_cuda, _ = batch2cuda(batch, config)
            logits = model(**batch_cuda)[3]
            probs = torch.sigmoid(logits)
            test_preds.append(probs.detach().cpu())

    ts = (time.time() - t) * 1000.0 / len(test_df) * 2000.0 / 7
    print(f'\n>>> Single action average cost time {ts:.4} ms on 2000 samples ...')

    test_preds = torch.cat(test_preds).numpy()
    test_df[ACTIONS] = test_preds
    test_df.drop('device', axis=1, inplace=True)
    submission_path = os.path.join(SUBMISSION_PATH, "nn{}.csv".format(config.seed))
    print(submission_path)
    test_df.to_csv(submission_path, index=False, encoding='utf8', sep=',')



def main():
    args = args_setup()

    collate_fn, processed_data = prepare_data(args)

    test_df = dt.fread(args.test_data_path).to_pandas()
    test_df['date_'] = 15
    if args.debug_data:
        test_df = test_df.head(1000)

    test_dataloader = DataLoader(WXDataset(test_df, processed_data), batch_size=args.bs, shuffle=False,
                                 collate_fn=collate_fn, num_workers=6, pin_memory=True)
    model = MultiDeepFM(args)
    model.to(args.device)
    args.best_model_path = os.path.join(args.output_dir, 'best.pth')
    model.load_state_dict(torch.load(args.best_model_path))
    
    predict(args, model, test_dataloader, test_df)

if __name__ == '__main__':
    main()
