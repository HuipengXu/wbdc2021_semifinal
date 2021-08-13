import os
import json
import math
import random
import pickle5 as pk5
from argparse import Namespace
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
import datatable as dt
from numba import njit
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer
from torch.utils.data import Sampler, BatchSampler, SubsetRandomSampler, Dataset

from ..common_path import *

# 每个行为的负样本下采样比例(下采样后负样本数/原负样本数)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


@njit
def auc(actual, pred_ranks):
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(pred_ranks[actual == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fast_auc(actual, predicted):
    # https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/208031
    pred_ranks = rankdata(predicted)
    return auc(actual, pred_ranks)


def roc_score(labels: list, preds: list,
              user_ids: list) -> dict:
    """多任务学习的 uauc
    read_comment, like, click_avatar, forward
    """
    task2weight = {'read_comment': 4, 'like': 3,
                   'click_avatar': 2, 'forward': 1,
                   'favorite': 1, 'comment': 1, 'follow': 1}

    preds = torch.cat(preds, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    columns = [f'{task}_{type_}' for type_ in ['label', 'pred'] for task in task2weight.keys()]
    results_df = pd.DataFrame(data=np.concatenate([labels, preds], axis=-1),
                              columns=columns)
    results_df.insert(0, column='user_id', value=user_ids)
    task_metrics = defaultdict(list)
    for _, data in results_df.groupby(by='user_id'):
        for task in task2weight.keys():
            labels = data.loc[:, f'{task}_label'].values
            preds = data.loc[:, f'{task}_pred'].values
            if len(np.unique(labels)) == 1:
                continue
            roc_auc = fast_auc(labels, preds)
            task_metrics[task].append(roc_auc)

    metrics = defaultdict(float)
    avg = 0.
    for task, rocs in task_metrics.items():
        task_roc = sum(rocs) / len(rocs)
        avg += task2weight[task] * task_roc
        metrics[task] = task_roc

    for metric in task2weight.keys():
        if metric not in metrics:
            metrics[metric] = 0.

    avg /= 13
    metrics['avg_roc'] = avg
    return metrics


def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)


def load_embedding_matrix(filepath='', max_vocab_size=50000):
    embedding_matrix = np.load(filepath)
    flag_matrix = np.zeros_like(embedding_matrix[:2])
    return np.concatenate([flag_matrix, embedding_matrix])[:max_vocab_size]


class Vocab:
    PAD_TOKEN = '<PAD>'
    UNKNOWN_TOKEN = '<UNK>'
    MASKS = [PAD_TOKEN, UNKNOWN_TOKEN]
    MASK_COUNT = len(MASKS)

    PAD_TOKEN_INDEX = MASKS.index(PAD_TOKEN)
    UNKNOWN_TOKEN_INDEX = MASKS.index(UNKNOWN_TOKEN)

    def __init__(self, vocab_file, max_vocab_size=None):
        """Vocab 对象,vocab基本操作封装
        :param vocab_file: Vocab 存储路径
        :param max_vocab_size: 最大字典数量
        """
        self.word2id, self.id2word = self.load_vocab(vocab_file, max_vocab_size)
        self.count = len(self.word2id)

    @staticmethod
    def load_vocab(file_path, vocab_max_size=None):
        """读取字典
        :param file_path: 文件路径
        :param vocab_max_size:
        :return: 返回读取后的字典
        """
        vocab = {mask: index
                 for index, mask in enumerate(Vocab.MASKS)}

        reverse_vocab = {index: mask
                         for index, mask in enumerate(Vocab.MASKS)}

        if isinstance(file_path, str):
            token2id = json.load(open(file_path, 'r', encoding='utf8'))
        else:
            token2id = file_path

        for word, index in token2id.items():
            index = int(index)
            # 如果vocab 超过了指定大小
            # 跳出循环 截断
            if vocab_max_size and index >= vocab_max_size - Vocab.MASK_COUNT:
                print("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (
                    vocab_max_size, index))
                break
            vocab[word] = index + Vocab.MASK_COUNT
            reverse_vocab[index + Vocab.MASK_COUNT] = word
        return vocab, reverse_vocab

    def word_to_id(self, word):
        if word not in self.word2id:
            return self.word2id[self.UNKNOWN_TOKEN]
        return self.word2id[word]

    def id_to_word(self, word_id):
        if word_id not in self.id2word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self.id2word[word_id]

    def size(self):
        return self.count


class SortedSampler(Sampler):
    """ Samples elements sequentially, always in the same order.

    Args:
        data (iterable): Iterable data.
        sort_key (callable): Specifies a function of one argument that is used to extract a
            numerical comparison key from each list element.

    Example:
        >>> list(SortedSampler(range(10), sort_key=lambda i: -i))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    """

    def __init__(self, data, sort_key):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)


class BucketBatchSampler(BatchSampler):
    """ `BucketBatchSampler` toggles between `sampler` batches and sorted batches.

    Typically, the `sampler` will be a `RandomSampler` allowing the user to toggle between
    random batches and sorted batches. A larger `bucket_size_multiplier` is more sorted and vice
    versa.

    Background:
        ``BucketBatchSampler`` is similar to a ``BucketIterator`` found in popular libraries like
        ``AllenNLP`` and ``torchtext``. A ``BucketIterator`` pools together examples with a similar
        size length to reduce the padding required for each batch while maintaining some noise
        through bucketing.

        **AllenNLP Implementation:**
        https://github.com/allenai/allennlp/blob/master/allennlp/data/iterators/bucket_iterator.py

        **torchtext Implementation:**
        https://github.com/pytorch/text/blob/master/torchtext/data/iterator.py#L225

    Args:
        sampler (torch.data.utils.sampler.Sampler):
        batch_size (int): Size of mini-batch.
        drop_last (bool): If `True` the sampler will drop the last batch if its size would be less
            than `batch_size`.
        sort_key (callable, optional): Callable to specify a comparison key for sorting.
        bucket_size_multiplier (int, optional): Buckets are of size
            `batch_size * bucket_size_multiplier`.
    """

    def __init__(self,
                 sampler,
                 batch_size,
                 drop_last,
                 sort_key,
                 bucket_size_multiplier=100):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.bucket_sampler = BatchSampler(sampler,
                                           min(batch_size * bucket_size_multiplier, len(sampler)),
                                           False)

    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            for batch in SubsetRandomSampler(
                    list(BatchSampler(sorted_sampler, self.batch_size, self.drop_last))):
                yield [bucket[i] for i in batch]

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)


def sentence2ids(sentence, vocab, sep=' '):
    # 字符串切分成词
    words = sentence.split(sep)
    # 按照vocab的index进行转换         # 遇到未知词就填充unk的索引
    ids = [vocab.word2id[word] if word in vocab.word2id else vocab.UNKNOWN_TOKEN_INDEX for word in words]
    if not ids:
        ids = [vocab.PAD_TOKEN_INDEX]
    return ids


def feed2ids(feed_hist, feed, tokenizer):
    feed = str(feed)
    inputs = tokenizer.encode_plus(feed_hist, feed, add_special_tokens=True, return_token_type_ids=True,
                                   return_attention_mask=False)
    return inputs


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def batch2cuda(batch, config):
    batch_cuda = []
    user_id = batch.pop('user_ids')
    for k, v in batch.items():
        if isinstance(v, dict):
            v = {vk: vv.to(config.device) for vk, vv in v.items()}
        else:
            v = v.to(config.device)
        batch_cuda.append((k, v))
    batch_cuda = dict(batch_cuda)
    return batch_cuda, user_id


class ProcessedData:

    def __init__(self, args):
        self.feed_embedding = dt.fread(
            os.path.join(COMPETITION_DATA_PATH, 'feed_embeddings.csv')
        ).to_pandas().set_index('feedid').to_dict('index')
        self.feed_info = dt.fread(
            os.path.join(TRAIN_TEST_DATA_PATH, 'feed_info.csv')
        ).to_pandas().set_index('feedid').to_dict('index')

        # svd
        self.user_token2id = load_json(args.user_token2id_path)
        self.uid_fid_svd = np.load(args.uid_fid_vector_path)
        self.fid_svd = np.load(args.fid_vector_path).T
        self.feed_token2id = load_json(args.feed_token2id_path)

        self.author_token2id = load_json(args.author_token2id_path)
        self.uid_aid_svd = np.load(args.uid_aid_vector_path)
        self.aid_svd = np.load(args.aid_vector_path).T

        self.bgm_song_token2id = load_json(args.bgm_song_token2id_path)
        self.uid_bgm_song_id_svd = np.load(args.uid_bgm_song_id_vector_path)
        self.bgm_song_id_svd = np.load(args.bgm_song_id_vector_path).T

        self.bgm_singer_token2id = load_json(args.bgm_singer_token2id_path)
        self.uid_bgm_singer_id_svd = np.load(args.uid_bgm_singer_id_vector_path)
        self.bgm_singer_id_svd = np.load(args.bgm_singer_id_vector_path).T

        self.keyword_list_token2id = load_json(args.keyword_list_token2id_path)
        self.uid_keyword_list_svd = np.load(args.uid_keyword_list_vector_path)
        self.keyword_list_svd = np.load(args.keyword_list_vector_path).T

        self.tag_list_token2id = load_json(args.tag_list_token2id_path)
        self.uid_tag_list_svd = np.load(args.uid_tag_list_vector_path)
        self.tag_list_svd = np.load(args.tag_list_vector_path).T

        # deepwalk
        self.userid_feedid_userid_deepwalk = pk5.load(
            open(args.userid_feedid_userid_deepwalk, 'rb')
        )
        self.userid_feedid_feedid_deepwalk = pk5.load(
            open(args.userid_feedid_feedid_deepwalk, 'rb')
        )
        # self.userid_authorid_userid_deepwalk = pk5.load(
        #     open(args.userid_authorid_userid_deepwalk, 'rb')
        # )
        # self.userid_authorid_authorid_deepwalk = pk5.load(
        #     open(args.userid_authorid_authorid_deepwalk, 'rb')
        # )
        # self.userid_bgm_song_id_userid_deepwalk = pk5.load(
        #     open(args.userid_bgm_song_id_userid_deepwalk, 'rb')
        # )
        # self.userid_bgm_song_id_bgm_song_id_deepwalk = pk5.load(
        #     open(args.userid_bgm_song_id_bgm_song_id_deepwalk, 'rb')
        # )
        # self.userid_bgm_singer_id_userid_deepwalk = pk5.load(
        #     open(args.userid_bgm_singer_id_userid_deepwalk, 'rb')
        # )
        # self.userid_bgm_singer_id_bgm_singer_id_deepwalk = pk5.load(
        #     open(args.userid_bgm_singer_id_bgm_singer_id_deepwalk, 'rb')
        # )
        # self.userid_keyword_list_userid_deepwalk = pk5.load(
        #     open(args.userid_keyword_list_userid_deepwalk, 'rb')
        # )
        # self.userid_keyword_list_keyword_list_deepwalk = pk5.load(
        #     open(args.userid_keyword_list_keyword_list_deepwalk, 'rb')
        # )
        # self.userid_tag_list_userid_deepwalk = pk5.load(
        #     open(args.userid_tag_list_userid_deepwalk, 'rb')
        # )
        # self.userid_tag_list_tag_list_deepwalk = pk5.load(
        #     open(args.userid_tag_list_tag_list_deepwalk, 'rb')
        # )

        # text
        self.vocab = Vocab(args.vocab_path, max_vocab_size=args.vocab_size)
        self.char_vocab = Vocab(args.char_vocab_path, max_vocab_size=args.char_vocab_size)
        self.device_vocab = {'1': 0, '2': 1}
        self.keyword_vocab = Vocab(args.keyword_vocab_path, max_vocab_size=args.keyword_vocab_size)
        self.tag_vocab = Vocab(args.tag_vocab_path, max_vocab_size=args.tag_vocab_size)


#         self.feed_tokenizer = BertTokenizer.from_pretrained('./data/uid_fid_svd/feed_vocab.txt')


class WXDataset(Dataset):

    def __init__(self, data_df: pd.DataFrame, processed_data: ProcessedData):
        self.data_df = data_df.to_numpy()
        self.column_index_dictionary = dict(zip(data_df.columns, list(range(len(data_df.columns)))))
        self.processed_data = processed_data

    def __getitem__(self, index):
        row = self.data_df[index]
        self.feedid = int(row[self.column_index_dictionary['feedid']])
        self.feed = self.processed_data.feed_info[self.feedid]
        (author_id_token_id, bgm_singer_id_token_id, bgm_song_id_token_id, device_id,
         feed_id_token_id, user_id_token_id) = self.get_sparse_feature_id(row)
        # text feature,
        (asr_char_ids, asr_ids, desc_char_ids, desc_ids, keyword_ids, ocr_char_ids,
         ocr_ids, tag_ids) = self.convert_token_to_sequence_ids()

        # 512
        feed_embedding = self.get_multi_modal_feed_emb()
        # feed_hist_embedding = self.get_feed_hist_embedding(row)
        #         feed_hist_ids = feed2ids(row[self.column_index_dictionary['feed_hist']],
        #                                  row[self.column_index_dictionary['feedid']],
        #                                  self.processed_data.feed_tokenizer)

        svd_features = np.concatenate(
            # 64
            [self.processed_data.uid_fid_svd[user_id_token_id],
             self.processed_data.fid_svd[feed_id_token_id],
             # 16
             # self.processed_data.uid_aid_svd[user_id_token_id],
             # self.processed_data.aid_svd[author_id_token_id],
             # self.processed_data.uid_bgm_song_id_svd[user_id_token_id],
             # self.processed_data.bgm_song_id_svd[bgm_song_id_token_id],
             # self.processed_data.uid_bgm_singer_id_svd[user_id_token_id],
             # self.processed_data.bgm_singer_id_svd[bgm_singer_id_token_id],
             # self.processed_data.uid_keyword_list_svd[user_id_token_id],
             # self.get_kw_tag_features(row, 'keyword_list', type_='svd'),
             # self.processed_data.uid_tag_list_svd[user_id_token_id],
             # self.get_kw_tag_features(row, 'tag_list', type_='svd')
             ],
            axis=-1
        )
        deepwalk_features = np.concatenate(
            [  # 64
                self.processed_data.userid_feedid_userid_deepwalk.get(int(row[self.column_index_dictionary['userid']]), np.zeros(64, dtype=np.float)),
                self.processed_data.userid_feedid_feedid_deepwalk.get(self.feedid, np.zeros(64, dtype=np.float)),
                # self.processed_data.userid_authorid_userid_deepwalk[row[self.column_index_dictionary['userid']]],
                # self.processed_data.userid_authorid_authorid_deepwalk[row[self.column_index_dictionary['authorid']]],
                # self.processed_data.userid_bgm_song_id_userid_deepwalk[row[self.column_index_dictionary['userid']]],
                # self.processed_data.userid_bgm_song_id_bgm_song_id_deepwalk[
                #    row[self.column_index_dictionary['bgm_song_id']]
                # ],
                # self.processed_data.userid_bgm_singer_id_userid_deepwalk[row[self.column_index_dictionary['userid']]],
                # self.processed_data.userid_bgm_singer_id_bgm_singer_id_deepwalk[
                #     row[self.column_index_dictionary['bgm_singer_id']]
                # ],
                # self.processed_data.userid_keyword_list_userid_deepwalk[row[self.column_index_dictionary['userid']]],
                # self.get_kw_tag_features(row, 'keyword_list'),
                # self.processed_data.userid_tag_list_userid_deepwalk[row[self.column_index_dictionary['userid']]],
                # self.get_kw_tag_features(row, 'tag_list')
            ],
            axis=-1
        )
        sparse_features = {
            'userid': user_id_token_id,
            'feedid': feed_id_token_id,
            'authorid': author_id_token_id,
            'device': device_id,
            'bgm_song_id': bgm_song_id_token_id,
            'bgm_singer_id': bgm_singer_id_token_id,
            'keyword_list': keyword_ids,
            'tag_list': tag_ids
        }

        if 'read_comment' in self.column_index_dictionary:
            labels = {
                'read_comment': row[self.column_index_dictionary['read_comment']],
                'like': row[self.column_index_dictionary['like']],
                'click_avatar': row[self.column_index_dictionary['click_avatar']],
                'forward': row[self.column_index_dictionary['forward']],
                'favorite': row[self.column_index_dictionary['favorite']],
                'comment': row[self.column_index_dictionary['comment']],
                'follow': row[self.column_index_dictionary['follow']],
            }
        else:
            labels = {
                'read_comment': 0,
                'like': 0,
                'click_avatar': 0,
                'forward': 0,
                'favorite': 0,
                'comment': 0,
                'follow': 0,
            }

        return (
            row[self.column_index_dictionary['userid']],
            svd_features,
            deepwalk_features,
            self.feed['videoplayseconds'], 
            sparse_features,
            feed_embedding,
            float(row[self.column_index_dictionary['date_']]),
            #             feed_hist_ids,
            # feed_hist_embedding,
            # ocr_ids,
            # ocr_char_ids,
            # asr_ids,
            # asr_char_ids,
            # desc_ids,
            # desc_char_ids,
            labels
        )

    def get_kw_tag_features(self, row, col_name, type_='deepwalk'):
        vocab = None
        if type_ == 'deepwalk':
            vecs = getattr(self.processed_data, f'userid_{col_name}_{col_name}_deepwalk')
        else:
            vecs = getattr(self.processed_data, f'{col_name}_svd')
            vocab = getattr(self.processed_data, f'{col_name}_token2id')
        item = row[self.column_index_dictionary[col_name]]
        features = np.zeros(64 if type_ == 'deepwalk' else 16, dtype=np.float)
        for token in item.split(';'):
            if (type_ == 'deepwalk' and token not in vecs) or (type_ == 'svd' and token not in vocab):
                continue
            if type_ == 'svd':
                token = vocab[token]
            features += vecs[token]
        return features.tolist()

    def get_multi_modal_feed_emb(self):
        feed_embedding = self.processed_data.feed_embedding[self.feedid]['feed_embedding']
        feed_embedding = np.fromstring(feed_embedding, dtype=float, sep=' ')
        return feed_embedding

    def get_feed_hist_embedding(self, row):
        feed_hists = []
        for feed_id in row[self.column_index_dictionary['feed_hist']].split()[:200]:
            feed_embedding = self.processed_data.feed_embedding[int(feed_id)]['feed_embedding']
            feed_embedding = np.fromstring(feed_embedding, dtype=float, sep=' ')
            feed_hists.append(feed_embedding)
        if not feed_hists:
            feed_hists.append(np.zeros(512, dtype=np.float))
        return feed_hists

    def convert_token_to_sequence_ids(self):
        desc_ids = sentence2ids(self.feed['description'], self.processed_data.vocab, sep=' ')
        ocr_ids = sentence2ids(self.feed['ocr'], self.processed_data.vocab, sep=' ')
        asr_ids = sentence2ids(self.feed['asr'], self.processed_data.vocab, sep=' ')
        desc_char_ids = sentence2ids(self.feed['description_char'], self.processed_data.char_vocab, sep=' ')
        ocr_char_ids = sentence2ids(self.feed['ocr_char'], self.processed_data.char_vocab, sep=' ')
        asr_char_ids = sentence2ids(self.feed['asr_char'], self.processed_data.char_vocab, sep=' ')
        keyword_ids = sentence2ids(self.feed['keyword_list'], self.processed_data.keyword_vocab, sep=';')
        tag_ids = sentence2ids(self.feed['tag_list'], self.processed_data.tag_vocab, sep=';')
        return asr_char_ids, asr_ids, desc_char_ids, desc_ids, keyword_ids, ocr_char_ids, ocr_ids, tag_ids

    def get_sparse_feature_id(self, row):
        user_id_token_id = self.processed_data.user_token2id[str(row[self.column_index_dictionary['userid']])]
        feed_id_token_id = self.processed_data.feed_token2id[str(self.feedid)]
        device_id = self.processed_data.device_vocab[str(row[self.column_index_dictionary['device']])]
        author_id_token_id = self.processed_data.author_token2id[str(self.feed['authorid'])]
        bgm_song_id_token_id = self.processed_data.bgm_song_token2id[str(int(self.feed['bgm_song_id']))]
        bgm_singer_id_token_id = self.processed_data.bgm_singer_token2id[str(int(self.feed['bgm_singer_id']))]
        return (author_id_token_id, bgm_singer_id_token_id, bgm_song_id_token_id,
                device_id, feed_id_token_id, user_id_token_id)

    def __len__(self):
        return self.data_df.shape[0]


class Collator:

    def __init__(self, args: Namespace):
        self.args = args

    #         self.feed_tokenizer = BertTokenizer.from_pretrained('./data/uid_fid_svd/feed_vocab.txt')

    @staticmethod
    def pad_and_truncate(data: list, max_len: int):
        tensor = torch.zeros((len(data), max_len), dtype=torch.long)
        for i, ex in enumerate(data):
            cur_len = len(ex)
            if cur_len > max_len:
                tensor[i] = torch.tensor(ex[:max_len], dtype=torch.long)
            else:
                tensor[i, :cur_len] = torch.tensor(ex, dtype=torch.long)
        return tensor

    @staticmethod
    def get_max_len(data):
        return max(len(sequence) for sequence in data)

    def process_feed_hist(self, feed_hists_list):
        max_len = max(len(input_dict['input_ids']) for input_dict in feed_hists_list)
        cur_max_len = min(max_len, 256)
        input_ids = torch.ones((len(feed_hists_list), cur_max_len),
                               dtype=torch.long) * self.feed_tokenizer.pad_token_id
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        for i, input_dict in enumerate(feed_hists_list):
            cur_len = len(input_dict['input_ids'])
            if cur_len <= cur_max_len:
                input_ids[i, :cur_len] = torch.tensor(input_dict['input_ids'], dtype=torch.long)
                token_type_ids[i, :cur_len] = torch.tensor(input_dict['token_type_ids'], dtype=torch.long)
                attention_mask[i, :cur_len] = 1
            else:
                input_ids[i] = torch.tensor(input_dict['input_ids'][:cur_max_len], dtype=torch.long)
                token_type_ids[i] = torch.tensor(input_dict['token_type_ids'][:cur_max_len], dtype=torch.long)
                attention_mask[i] = 1
        return {
            'input_ids': input_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }

    def __call__(self, examples: list):
        # (user_id_list, svd_list, deepwalk_list, dense_list, sparse_list, feed_emb_list, ocr_list, ocr_char_list,
        #   asr_list, asr_char_list, desc_list, desc_char_list, labels_list) = zip(*examples)
        (user_id_list, svd_list, deepwalk_list, dense_list, sparse_list, feed_emb_list, date_list,
         labels_list) = zip(*examples)
        svd_features = torch.tensor(svd_list, dtype=torch.float)
        deepwalk_features = torch.tensor(deepwalk_list, dtype=torch.float)

        dense_features = torch.tensor(dense_list, dtype=torch.float).view(-1, 1)
        date_features = torch.tensor(date_list, dtype=torch.float).view(-1, 1)

        sparse_features = self.process_sparse_features(sparse_list)

        feed_embedding = torch.tensor(feed_emb_list, dtype=torch.float)
        #         feed_hists_inputs = self.process_feed_hist(feed_hists_list)

        # padding and cut
        # asr, asr_char, desc, desc_char, ocr, ocr_char = self.process_text_features(asr_char_list, asr_list,
        #                                                                            desc_char_list, desc_list,
        #                                                                            ocr_char_list, ocr_list)
        labels = self.process_labels(labels_list)
        return {
            'user_ids': list(user_id_list),
            'svd_features': svd_features,
            'deepwalk_features': deepwalk_features,
            'dense_features': dense_features,
            'sparse_features': sparse_features,
            'feed_embedding': feed_embedding,
            'date': date_features,
            #             'feed_hists': feed_hists_inputs,
            # 'feed_hists_emb': feed_hists_embedding,
            # 'ocr': ocr,
            # 'ocr_char': ocr_char,
            # 'asr': asr,
            # 'asr_char': asr_char,
            # 'desc': desc,
            # 'desc_char': desc_char,
            'labels': labels
        }

    @staticmethod
    def process_labels(labels_list):
        labels = defaultdict(list)
        for act_dict in labels_list:
            for act, label in act_dict.items():
                labels[act].append(label)
        labels = {act: torch.tensor(labels, dtype=torch.float) for act, labels in labels.items()}
        return labels

    def process_text_features(self, asr_char_list, asr_list, desc_char_list,
                              desc_list, ocr_char_list, ocr_list):
        max_ocr_len = min(self.get_max_len(ocr_list), self.args.ocr_len)
        max_ocr_char_len = min(self.get_max_len(ocr_char_list), self.args.ocr_char_len)
        max_asr_len = min(self.get_max_len(asr_list), self.args.asr_len)
        max_asr_char_len = min(self.get_max_len(asr_char_list), self.args.asr_char_len)
        max_desc_len = min(self.get_max_len(desc_list), self.args.desc_len)
        max_desc_char_len = min(self.get_max_len(desc_char_list), self.args.desc_char_len)
        ocr = self.pad_and_truncate(ocr_list, max_ocr_len)
        ocr_char = self.pad_and_truncate(ocr_char_list, max_ocr_char_len)
        asr = self.pad_and_truncate(asr_list, max_asr_len)
        asr_char = self.pad_and_truncate(asr_char_list, max_asr_char_len)
        desc = self.pad_and_truncate(desc_list, max_desc_len)
        desc_char = self.pad_and_truncate(desc_char_list, max_desc_char_len)
        return asr, asr_char, desc, desc_char, ocr, ocr_char

    def process_sparse_features(self, sparse_list):
        sparse_features = defaultdict(list)
        for sf in sparse_list:
            for k, v in sf.items():
                sparse_features[k].append(v)
        max_num_kw = self.get_max_len(sparse_features['keyword_list'])
        max_num_tag = self.get_max_len(sparse_features['tag_list'])
        for k, v in sparse_features.items():
            if k == 'keyword_list':
                sparse_features[k] = self.pad_and_truncate(v, max_num_kw)
            elif k == 'tag_list':
                sparse_features[k] = self.pad_and_truncate(v, max_num_tag)
            else:
                sparse_features[k] = torch.tensor(v, dtype=torch.long)
        return sparse_features


def undersampling_neg_sample_all_actions(df: pd.DataFrame, args: Namespace):
    print(f'\n>>> Starting undersampling negative data ...')

    neg_df = df[df["action_num"] == 0]
    neg_df = neg_df.sample(frac=0.5,
                           random_state=args.seed, replace=False)
    pos_df = df[df["action_num"] > 0]
    action_df = pd.concat([neg_df, pos_df], axis=0)
    action_df = action_df.sample(frac=1, random_state=args.seed, replace=False)
    print(f'\n>>> After undersampling, pos : neg = {len(pos_df)} : {len(neg_df)} ...')
    print(f'\n>>> After undersampling, neg : pos ratios = {len(neg_df)}/{len(pos_df)} ...')
    num_samples = len(action_df)
    print(f'\n>>> total number is {num_samples} ...')
    return action_df


def read_data(merge_data_path, debug=False, mode='train'):
    data_df = dt.fread(merge_data_path).to_pandas()
    #     feed_hist = pd.read_csv(f'./data/{mode}_feed_hist.csv')
    #     data_df['feed_hist'] = feed_hist
    if debug:
        data_df = data_df.head(1000)
    convert_cols = data_df.select_dtypes(include=['int64', 'float64', 'int32', 'bool']).columns
    dtype_map = {col: 'int32' for col in convert_cols}
    data_df = data_df.astype(dtype_map)
    data_df.drop(['manual_keyword_list', 'machine_keyword_list',
                  'manual_tag_list', 'machine_tag_list'], axis=1, inplace=True)
    data_df.fillna('', inplace=True)
    data_df.videoplayseconds = np.log(data_df.videoplayseconds + 1.0)
    return data_df
