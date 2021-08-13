import os
from argparse import Namespace
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel, BertConfig
from torch.random import seed
from torch.nn.modules.activation import SELU
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from .activation import activation_layer


class DNN(nn.Module):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be
        a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input
        with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)
            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class CIN(nn.Module):
    """Compressed Interaction Network used in xDeepFM.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, featuremap_num)``
        ``featuremap_num =  sum(self.layer_size[:-1]) // 2 + self.layer_size[-1]``
        if ``split_half=True``,else  ``sum(layer_size)`` .
      Arguments
        - **filed_size** : Positive integer, number of feature groups.
        - **layer_size** : list of int.Feature maps in each layer.
        - **activation** : activation function name used on feature maps.
        - **split_half** : bool.if set to False, half of the feature maps in each hidden will connect to output unit.
        - **seed** : A Python integer to use as random seed.
      References
        - [Lian J, Zhou X, Zhang F, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for
        Recommender Systems[J]. arXiv preprint arXiv:1803.05170, 2018.] (https://arxiv.org/pdf/1803.05170.pdf)
    """

    def __init__(self, field_size, layer_size=(128, 128), activation='relu', split_half=True, l2_reg=1e-5, seed=1024,
                 device='cpu'):
        super(CIN, self).__init__()
        if len(layer_size) == 0:
            raise ValueError(
                "layer_size must be a list(tuple) of length greater than 1")

        self.layer_size = layer_size
        self.field_nums = [field_size]
        self.split_half = split_half
        self.activation = activation_layer(activation)
        self.l2_reg = l2_reg
        self.seed = seed

        self.conv1ds = nn.ModuleList()
        for i, size in enumerate(self.layer_size):
            self.conv1ds.append(
                nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))

            if self.split_half:
                if i != len(self.layer_size) - 1 and size % 2 > 0:
                    raise ValueError(
                        "layer_size must be even number except for the last layer when split_half=True")

                self.field_nums.append(size // 2)
            else:
                self.field_nums.append(size)

        #         for tensor in self.conv1ds:
        #             nn.init.normal_(tensor.weight, mean=0, std=init_std)
        self.to(device)

    def forward(self, inputs):
        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))
        batch_size = inputs.shape[0]
        dim = inputs.shape[-1]
        hidden_nn_layers = [inputs]
        final_result = []

        for i, size in enumerate(self.layer_size):
            # x^(k-1) * x^0
            x = torch.einsum(
                'bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            # x.shape = (batch_size , hi * m, dim)
            x = x.reshape(
                batch_size, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], dim)
            # x.shape = (batch_size , hi, dim)
            x = self.conv1ds[i](x)

            if self.activation is None or self.activation == 'linear':
                curr_out = x
            else:
                curr_out = self.activation(x)

            if self.split_half:
                if i != len(self.layer_size) - 1:
                    next_hidden, direct_connect = torch.split(
                        curr_out, 2 * [size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
            else:
                direct_connect = curr_out
                next_hidden = curr_out

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = torch.cat(final_result, dim=1)
        result = torch.sum(result, -1)

        return result


class FM(nn.Module):
    """Factorization Machine models pairwise (order-2) feature interactions
     without linear term and bias.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, 1)``.
      References
        - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
    """

    def __init__(self):
        super(FM, self).__init__()

    def forward(self, inputs):
        fm_input = inputs

        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)

        return cross_term


class Attention(nn.Module):
    def __init__(self, chans):
        super(Attention, self).__init__()
        self.attn = nn.Linear(chans, 1)

    def forward(self, emb):
        # n,k 10
        mean_emb = emb.mean(axis=1, keepdim=True)
        N, K, D = emb.size()
        att_emb = ((K - 1.0) * 1.0 / K) * emb + 1.0 / K * mean_emb
        attn_score = self.attn(att_emb)
        # maybe sigmmoid的
        attn_probs = torch.sigmoid(attn_score)
        attn_emb = (emb * attn_probs).sum(dim=1)
        return attn_emb


class AttentionForEmbedding(nn.Module):

    def __init__(self, config: Namespace):
        super(AttentionForEmbedding, self).__init__()
        self.config = config
        self.multi_modal_embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(config.multi_modal_emb_matrix),
            freeze=config.freeze_multi_modal_emb
        )
        self.multi_modal_embedding_char = nn.Embedding.from_pretrained(
            torch.from_numpy(config.multi_modal_emb_char_matrix),
            freeze=config.freeze_multi_modal_emb
        )
        self.attn = nn.Linear(config.multi_modal_emb_size, 1)
        self.char_attn = nn.Linear(config.multi_modal_emb_size, 1)
        self.multi_modal_attn = nn.Linear(config.multi_modal_emb_size, 1)

    @staticmethod
    def attention(attn_layer, emb_layer=None,
                  sequence_ids=None, emb=None):
        if emb is None:
            mask = sequence_ids == 0
            emb = emb_layer(sequence_ids)
        mean_emb = emb.mean(axis=1, keepdim=True)
        N, K, D = emb.size()
        att_emb = ((K - 1) / K) * emb + 1 / K * mean_emb
        attn_score = attn_layer(att_emb)
        if emb is None:
            attn_score.masked_fill_(mask, -1e9)
        # maybe sigmmoid的
        attn_probs = torch.sigmoid(attn_score)
        attn_emb = (emb * attn_probs).sum(dim=1)
        return attn_emb

    def forward(
            self,
            ocr: torch.Tensor = None,
            ocr_char: torch.Tensor = None,
            asr: torch.Tensor = None,
            asr_char: torch.Tensor = None,
            desc: torch.Tensor = None,
            desc_char: torch.Tensor = None,
    ):
        multi_modal_attn_emb = []
        # word
        for sequence_ids in [ocr, asr, desc]:
            multi_modal_attn_emb.append(
                self.attention(self.attn, self.multi_modal_embedding,
                               sequence_ids)
            )
        # char
        for sequence_ids in [ocr_char, asr_char, desc_char]:
            multi_modal_attn_emb.append(
                self.attention(self.char_attn, self.multi_modal_embedding_char,
                               sequence_ids)
            )
        # cls
        multi_modal_attn_emb = torch.stack(multi_modal_attn_emb, dim=1)
        multi_modal_attn_out = self.attention(
            self.multi_modal_attn, emb=multi_modal_attn_emb
        )
        return multi_modal_attn_out


def create_embedding_dict(config, linear=False):
    direct_emb = 4
    embedding_dict = nn.ModuleDict(
        {
            feat_name: nn.Embedding(feat_size, feat_emb_size if not linear else direct_emb)
            for (feat_name, feat_size, feat_emb_size)
            in config.sparse_feature_info
        }
    )
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=config.init_std)

    return embedding_dict

class LongShortMem(nn.Module):
    def __init__(self, config):
        super(LongShortMem, self).__init__()
        self.hid_dims = 64
        self.num_tasks = 7
        user_name, user_size, user_emb_size = config.sparse_feature_info[0]
        feed_name, feed_size, feed_emb_size = config.sparse_feature_info[1]
        
        self.shortlong_mem = nn.Parameter(torch.Tensor(user_size, self.hid_dims))
        
        self.use_learning1 = nn.Sequential(nn.Linear(user_emb_size+feed_emb_size+10, self.hid_dims),
                                        nn.GELU(),
                                        nn.Linear(self.hid_dims, self.hid_dims),
                                        nn.GELU(),
                                        nn.Linear(self.hid_dims, self.hid_dims),   
                                        nn.GELU(),
                                        nn.Linear(self.hid_dims, self.hid_dims))
        
        
        self.long_short_fc = nn.Sequential(nn.Linear(self.hid_dims + self.hid_dims, self.hid_dims),
                                           nn.GELU(),
                                           nn.Linear(self.hid_dims, self.num_tasks))
        
    
    def forward(self, date, feed_feat, user_feat, user_id):
        date_emb = [torch.cos(date*i*0.5) for i in torch.arange(1,6)] + [torch.cos(-date*i*0.5) for i in torch.arange(1,6)]
        date_emb = torch.cat(date_emb, 1)
        merge_feat = torch.cat([user_feat, feed_feat, date_emb], 1)
            
        #(batch, num_head)
        feat1 = self.use_learning1(merge_feat)
        hist_feat = self.shortlong_mem[user_id]
    
        short_long_logits = self.long_short_fc(torch.cat([feat1, hist_feat], 1))
        
        return short_long_logits
        

class Linear(nn.Module):

    def __init__(self, config):
        super(Linear, self).__init__()
        self.embedding_dict = create_embedding_dict(config, linear=False)
        self.weight = nn.Parameter(torch.Tensor(1, 7))
        self.trans_weight = nn.Parameter(torch.Tensor(80, 7))
        self.attent_tag = Attention(10)
        self.attent_key = Attention(10)
        torch.nn.init.normal_(self.weight, mean=0, std=config.init_std)
        torch.nn.init.normal_(self.trans_weight, mean=0, std=config.init_std)

    def forward(self,
                dense_features: torch.Tensor,
                sparse_features: Dict[str, torch.Tensor],
                ):
        sparse_embeddings_dict = {feat_name: self.embedding_dict[feat_name](sparse_idx)
                                  for feat_name, sparse_idx in sparse_features.items()}
        # sparse_embeddings_dict['keyword_list'] = sparse_embeddings_dict['keyword_list'].sum(dim=1)
        # sparse_embeddings_dict['tag_list'] = sparse_embeddings_dict['tag_list'].sum(dim=1)
        # N, k, 10
        sparse_embeddings_dict['keyword_list'] = self.attent_key(sparse_embeddings_dict['keyword_list'])
        sparse_embeddings_dict['tag_list'] = self.attent_tag(sparse_embeddings_dict['tag_list'])
        lin_embedding = torch.cat(list(sparse_embeddings_dict.values()), dim=-1)
        lin_logits = (lin_embedding @ self.trans_weight)
        dense_logits = (dense_features @ self.weight)
        return lin_logits + dense_logits


class BertForFeed(nn.Module):

    def __init__(self):
        super(BertForFeed, self).__init__()
        self.config = BertConfig()
        self.config.vocab_size = 106449
        self.config.hidden_size = 128
        self.config.num_hidden_layers = 3
        self.config.num_attention_heads = 4
        self.config.intermediate_size = 256
        self.config.max_position_embeddings = 256
        self.bert = BertModel(self.config, add_pooling_layer=False)

    def forward(self, feed_hists):
        outputs = self.bert(**feed_hists)
        hidden_states, _ = torch.max(outputs[0], dim=1)
        return hidden_states


class Expert(nn.Module):

    def __init__(self, config: Namespace):
        super(Expert, self).__init__()
        self.config = config

        self.embedding_dict = create_embedding_dict(config)
        self.key_attent = Attention(10)
        self.tag_attent = Attention(10)

        # self.fm = FM()
        self.cin = CIN(field_size=config.num_field,
                       layer_size=config.cin_layer_size,
                       activation=config.cin_activation,
                       split_half=config.cin_split_half,
                       l2_reg=config.l2_reg_cin,
                       seed=config.seed,
                       device=config.device)
        if config.cin_split_half:
            self.num_feature_map = sum(config.cin_layer_size[:-1]) // 2 + config.cin_layer_size[-1]
        else:
            self.num_feature_map = sum(config.cin_layer_size)
        self.dense_bn = nn.BatchNorm1d(config.dnn_inputs_dim)
        self.dnn = DNN(inputs_dim=config.dnn_inputs_dim,
                       hidden_units=config.dnn_hidden_units,
                       activation=config.dnn_activation,
                       l2_reg=config.l2_reg_dnn,
                       dropout_rate=config.dnn_dropout,
                       use_bn=config.dnn_use_bn,
                       init_std=config.init_std,
                       seed=config.seed,
                       device=config.device)
        # self.rnn_encoder = RNNEncoder(config)
        # self.attn_emb = AttentionForEmbedding(config)

        self.attent_layer = nn.Sequential(nn.BatchNorm1d(256 + 128),
                                          nn.Linear(256 + 128, 32),
                                          nn.BatchNorm1d(32),
                                          nn.ReLU(),
                                          nn.Linear(32, 32),
                                          nn.BatchNorm1d(32),
                                          nn.ReLU(),
                                          nn.Linear(32, 256 + 128),
                                          nn.Sigmoid())

    def forward(
            self,
            svd_features: torch.Tensor,
            deepwalk_features: torch.Tensor,
            dense_features: torch.Tensor,
            sparse_features: Dict[str, torch.Tensor],
            feed_embedding: torch.Tensor,
            ocr: torch.Tensor = None,
            ocr_char: torch.Tensor = None,
            asr: torch.Tensor = None,
            asr_char: torch.Tensor = None,
            desc: torch.Tensor = None,
            desc_char: torch.Tensor = None
    ):
        """
        :param svd_features: (bs, 224)
        :param deepwalk_features: (bs, 512)
        :param dense_features: (bs, num_dense(1-feed时长))
        :param sparse_features: e.g. {'userid': (bs, )}
        :param feed_embedding: (bs, 512)
        :param ocr: (bs, ocr_len)
        :param ocr_char: (bs, ocr_char_len)
        :param asr: (bs, asr_len)
        :param asr_char: (bs, asr_char_len)
        :param desc: (bs, desc_len)
        :param desc_char: (bs, desc_char_len)
        :param labels: (bs, 4)
        :return:
        """
        # sparse_features.pop('keyword_id')
        # sparse_features.pop('tag_id')
        # sparse_features.pop('device_id')
        sparse_embeddings_dict = {feat_name: self.embedding_dict[feat_name](sparse_idx)
                                  for feat_name, sparse_idx in sparse_features.items()}
        # 多个key，sum
        sparse_embeddings_dict['keyword_list'] = self.key_attent(sparse_embeddings_dict['keyword_list'])
        sparse_embeddings_dict['tag_list'] = self.tag_attent(sparse_embeddings_dict['tag_list'])
        sparse_embeddings_list = list(sparse_embeddings_dict.values())

        # 稀疏特征的组合
        sparse_embedding = torch.stack(sparse_embeddings_list, dim=1)  # (bs, num_field, emb_size)
        # fm_outputs = self.fm(sparse_embedding)
        cin_feat = self.cin(sparse_embedding)

        # (bs, 200)
        # multi_modal_features = self.attn_emb(ocr, ocr_char, asr,
        # asr_char, desc, desc_char)
        # dense_inputs = torch.cat([dense_features, multi_modal_features] + sparse_embeddings_list, dim=-1)
        dense_inputs = torch.cat([deepwalk_features, dense_features, svd_features] + sparse_embeddings_list, dim=-1)
        dense_inputs = self.dense_bn(dense_inputs)
        dnn_feat = self.dnn(dense_inputs)
        feats = torch.cat([cin_feat, dnn_feat], 1)
        # feats = feats * self.attent_layer(feats)
        feats = self.attent_layer(feats) # need vrify.
        return feats


class TaskTower(nn.Module):
    def __init__(self, in_chans):
        super(TaskTower, self).__init__()
        self.dnn_logits = nn.Sequential(
            nn.BatchNorm1d(in_chans),
            nn.Linear(in_chans, 128),
            nn.GELU(),
            nn.Linear(128, 1))

    def forward(self, dnn_feat):
        dnn_logits = self.dnn_logits(dnn_feat)
        return dnn_logits

class MutliHead(nn.Module):
    def __init__(self, num_head,num_task, config: Namespace):
        super(MutliHead, self).__init__()
        self.share_feed = nn.Linear(512 + config.dnn_hidden_units[-1] + 256, 256)
        self.num_head = num_head
        self.num_task = num_task
        self.feed_list = nn.ModuleList([nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64)) for i in range(self.num_head)])
        self.attent_layer = nn.Sequential(nn.Linear(512 + config.dnn_hidden_units[-1] + 256, 32),
                                          nn.BatchNorm1d(32),
                                          nn.ReLU(),
                                          nn.Linear(32, 32),
                                          nn.BatchNorm1d(32),
                                          nn.ReLU(),
                                          nn.Linear(32, self.num_head*64))
        self.final_fc = nn.Linear(64, self.num_task)
    def forward(self, x):
        shared_x = self.share_feed(x)
        head_result = []
        for i in range(self.num_head):
            head_result.append(self.feed_list[i](shared_x))
        multi_feat = torch.stack(head_result, 1)
        multi_att = self.attent_layer(x).reshape((-1, self.num_head, 64))
        multi_att = F.softmax(multi_att, 1)
        multi_feat = multi_feat * multi_att
        multi_feat = multi_feat.sum(1)
        multi_result = self.final_fc(multi_feat)
        return multi_result

class MultiDeepFM(nn.Module):

    def __init__(self, config: Namespace):
        super(MultiDeepFM, self).__init__()
        self.config = config
        self.num_expert = 3
        self.task_num = 7
        self.key_attent = Attention(10)
        self.tag_attent = Attention(10)
#         self.feed_bert = BertForFeed()
        # self.attn_emb = AttentionForEmbedding(config)
        self.embedding_dict = create_embedding_dict(config)
        # feed embeding and dnn fuse
        self.feed_bn = nn.BatchNorm1d(512 + config.dnn_hidden_units[-1] + 256)
        self.feed_fc = nn.Sequential(
            nn.Linear(512 + config.dnn_hidden_units[-1] + 256, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.task_num)
        )
        # self.feed_fc = MutliHead(3, self.task_num, config)
        self.long_short_mem = LongShortMem(config)
        self.lin = Linear(config)
        self.expert_list = nn.ModuleList([Expert(self.config) for i in range(self.num_expert)])
        self.input_size = self.config.dnn_inputs_dim
        self.gates_inear = nn.ModuleList(
            [nn.Linear(self.input_size, self.num_expert) for i in range(self.task_num)])
        self.in_chans = self.config.dnn_hidden_units[-1] + self.config.cin_layer_size[-1] * 2
        self.tower_list = nn.ModuleList([TaskTower(self.in_chans) for i in range(self.task_num)])

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')

    def forward(
            self,
            svd_features: torch.Tensor,
            deepwalk_features: torch.Tensor,
            dense_features: torch.Tensor,
            sparse_features: Dict[str, torch.Tensor],
            feed_embedding: torch.Tensor,
            date: torch.Tensor,
#             feed_hists: Dict[str, torch.Tensor],
#             ocr: torch.Tensor = None,
#             ocr_char: torch.Tensor = None,
#             asr: torch.Tensor = None,
#             asr_char: torch.Tensor = None,
#             desc: torch.Tensor = None,
#             desc_char: torch.Tensor = None,
            labels: torch.Tensor = None
    ):
        # sparse_features.pop('keyword_id')
        # sparse_features.pop('tag_id')
        # sparse_features.pop('device_id')
        sparse_embeddings_dict = {feat_name: self.embedding_dict[feat_name](sparse_idx)
                                  for feat_name, sparse_idx in sparse_features.items()}
        # 多个key，sum
        sparse_embeddings_dict['keyword_list'] = self.key_attent(sparse_embeddings_dict['keyword_list'])
        sparse_embeddings_dict['tag_list'] = self.tag_attent(sparse_embeddings_dict['tag_list'])
        sparse_embeddings_list = list(sparse_embeddings_dict.values())

        # (bs, 200)
        # multi_modal_features = self.attn_emb(ocr, ocr_char, asr,
        #                                      asr_char, desc, desc_char)

        # dense_inputs = torch.cat([dense_features, multi_modal_features] + sparse_embeddings_list, dim=-1)
        gate_inputs = torch.cat([deepwalk_features, dense_features, svd_features] + sparse_embeddings_list, dim=-1)
        self.expert_feat = []
        for layer in self.expert_list:
            self.expert_feat.append(layer(svd_features, deepwalk_features, dense_features, sparse_features,
                                          feed_embedding))
        self.expert_feat = torch.stack(self.expert_feat, 1)
        long_short_logits = self.long_short_mem(date, 
                                               self.embedding_dict['feedid'](sparse_features["feedid"]),
                                               self.embedding_dict['userid'](sparse_features["userid"]),
                                               sparse_features["userid"])
        task_logits = []
        for i in range(self.task_num):
            attent = self.gates_inear[i](gate_inputs)
            attent = F.softmax(attent, dim=-1).unsqueeze(-1)
            task_input = (attent * self.expert_feat).sum(1)
            task_logits.append(self.tower_list[i](task_input))

#         feed_hists_state = self.feed_bert(feed_hists)

        labels = torch.stack(list(labels.values()), dim=1)
        deep_logits = torch.cat(task_logits, 1)
        merge_feed = torch.cat([feed_embedding, self.expert_feat.sum(1)], dim=1)
        merge_feed = self.feed_bn(merge_feed)
        feed_logits = self.feed_fc(merge_feed)
        lin_logits = self.lin(dense_features, sparse_features)
        
                                      
        logits = lin_logits + feed_logits + deep_logits + long_short_logits
        loss = self.loss_fct(logits, labels)
        pos_loss = loss[labels == 1].sum()
        neg_loss = loss[labels == 0].sum()
        loss = loss.sum()
#         nums_loss = torch.tensor(0.)
#         nums_logits = torch.zeros((labels.size(0), 1))

        return pos_loss, neg_loss, loss, logits, labels


class TestMultiDeepFM(nn.Module):

    def __init__(self, config):
        super(TestMultiDeepFM, self).__init__()
        self.config = config
        self.num_expert = 3
        self.task_num = 7
        self.key_attent = Attention(10)
        self.tag_attent = Attention(10)
        self.embedding_dict = create_embedding_dict(config)
        # feed embeding and dnn fuse
        self.feed_bn = nn.BatchNorm1d(512 + config.dnn_hidden_units[-1] + 256)
        self.feed_fc = nn.Sequential(
            nn.Linear(512 + config.dnn_hidden_units[-1] + 256, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.task_num)
        )
        self.mem_fc = ()
        self.lin = Linear(config)
        self.expert_list = nn.ModuleList([Expert(self.config) for i in range(self.num_expert)])
        self.input_size = self.config.dnn_inputs_dim
        self.gates_inear = nn.ModuleList(
            [nn.Linear(self.input_size, self.num_expert) for i in range(self.task_num)])
        self.in_chans = self.config.dnn_hidden_units[-1] + self.config.cin_layer_size[-1] * 2 + 128
        self.tower_list = nn.ModuleList([TaskTower(self.in_chans) for i in range(self.task_num)])

#         self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    def gather_args(self, args, kw):
        features = {}
        for arg, val in args.items():
            if kw in arg:
                name = '_'.join(arg.split('_')[:-1])
                features[name] = val
        return features
                  
    def forward(
            self,
            svd_features: torch.Tensor,
            deepwalk_features: torch.Tensor,
            dense_features: torch.Tensor,
            userid_sparse: torch.Tensor,
            feedid_sparse: torch.Tensor,
            authorid_sparse: torch.Tensor,
            device_sparse: torch.Tensor,
            bgm_song_id_sparse: torch.Tensor,
            bgm_singer_id_sparse: torch.Tensor,
            keyword_list_sparse: torch.Tensor,
            tag_list_sparse: torch.Tensor,
            feed_embedding: torch.Tensor,
#             read_comment_labels: torch.Tensor,
#             like_labels: torch.Tensor,
#             click_avatar_labels: torch.Tensor,
#             forward_labels: torch.Tensor
    ):
        args = locals()
        sparse_features = self.gather_args(args, 'sparse')
#         labels = self.gather_args(args, 'labels')
        sparse_embeddings_dict = {feat_name: self.embedding_dict[feat_name](sparse_idx)
                                  for feat_name, sparse_idx in sparse_features.items()}
        # 多个key，sum
        sparse_embeddings_dict['keyword_list'] = self.key_attent(sparse_embeddings_dict['keyword_list'])
        sparse_embeddings_dict['tag_list'] = self.tag_attent(sparse_embeddings_dict['tag_list'])
        sparse_embeddings_list = list(sparse_embeddings_dict.values())

        # (bs, 200)
        # multi_modal_features = self.attn_emb(ocr, ocr_char, asr,
        #                                      asr_char, desc, desc_char)

        # dense_inputs = torch.cat([dense_features, multi_modal_features] + sparse_embeddings_list, dim=-1)
        gate_inputs = torch.cat([deepwalk_features, dense_features, svd_features] + sparse_embeddings_list, dim=-1)
        self.expert_feat = []
        for layer in self.expert_list:
            self.expert_feat.append(layer(svd_features, deepwalk_features, dense_features, sparse_features,
                                          feed_embedding))
        self.expert_feat = torch.stack(self.expert_feat, 1)
        task_logits = []
        for i in range(self.task_num):
            attent = self.gates_inear[i](gate_inputs)
            attent = F.softmax(attent, dim=-1).unsqueeze(-1)
            task_input = (attent * self.expert_feat).sum(1)
            task_logits.append(self.tower_list[i](task_input))

#         feed_hists_state = self.feed_bert(feed_hists)

#         labels = torch.stack(list(labels.values()), dim=1)
        deep_logits = torch.cat(task_logits, 1)
        merge_feed = torch.cat([feed_embedding, self.expert_feat.sum(1)], dim=1)
        merge_feed = self.feed_bn(merge_feed)
        feed_logits = self.feed_fc(merge_feed)
        lin_logits = self.lin(dense_features, sparse_features)
        logits = lin_logits + feed_logits + deep_logits
#         logits1 = torch.cat([1-logits[:, [0]], logits[:, [0]]], dim=1)
#         logits2 = torch.cat([1-logits[:, [1]], logits[:, [1]]], dim=1)
#         logits3 = torch.cat([1-logits[:, [2]], logits[:, [2]]], dim=1)
#         logits4 = torch.cat([1-logits[:, [3]], logits[:, [3]]], dim=1)
#         loss = torch.stack((self.loss_fct(logits1, labels[:, 0].long()), self.loss_fct(logits1, labels[:, 1].long()), 
#                 self.loss_fct(logits1, labels[:, 2].long()), self.loss_fct(logits1, labels[:, 3].long())), dim=1)
#         pos_loss = loss[labels == 1].sum()
#         neg_loss = loss[labels == 0].sum()
#         loss = loss.sum()
# #         nums_loss = torch.tensor(0.)
# #         nums_logits = torch.zeros((labels.size(0), 1))

        return logits

