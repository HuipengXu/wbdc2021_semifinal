import os
import argparse
import numpy as np
from ..common_path import *

def get_args():
    parser = argparse.ArgumentParser(description='XDeepFM')

    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--ocr_char_len', type=int, default=128,
                        help='quantile 0.97')
    parser.add_argument('--ocr_len', type=int, default=64,
                        help='quantile 0.97')
    parser.add_argument('--desc_len', type=int, default=98,
                        help='quantile 0.99')
    parser.add_argument('--desc_char_len', type=int, default=123,
                        help='quantile 0.99')
    parser.add_argument('--asr_len', type=int, default=337,
                        help='quantile 0.97')
    parser.add_argument('--asr_char_len', type=int, default=128,
                        help='quantile 0.97')

    ########################## XDeepFM ##########################
    parser.add_argument('--multi_modal_hidden_size', type=int, default=128)
    parser.add_argument('--num_field', type=int, default=8)
    parser.add_argument('--cin_layer_size', type=tuple, default=(256, 128))
    parser.add_argument('--cin_split_half', type=bool, default=True)
    parser.add_argument('--cin_activation', type=str, default='relu')
    parser.add_argument('--dnn_activation', type=str, default='relu')
    parser.add_argument('--l2_reg_cin', type=float, default=0.)
    parser.add_argument('--l2_reg_dnn', type=float, default=0.)
    parser.add_argument('--dnn_use_bn', type=bool, default=True)
    parser.add_argument('--init_std', type=float, default=0.0001)
    parser.add_argument('--dnn_dropout', type=float, default=0.)
    parser.add_argument('--dnn_hidden_units', type=tuple, default=(256, 128, 128))
    parser.add_argument('--dnn_inputs_dim', type=int, default=337+3)
    parser.add_argument('--l2_reg_embedding', type=float, default=1e-5)
    parser.add_argument('--l2_reg_linear', type=float, default=1e-5)
    parser.add_argument('--l2', type=float, default=1e-3)
    ########################## XDeepFM ##########################
    parser.add_argument('--sparse_feature_info', type=list,
                        default=[('userid', 220000, 10), ('feedid', 106444, 10),
                                 ('authorid', 18789, 10), ('device', 2, 10),
                                 ('bgm_song_id', 25160, 10), ('bgm_singer_id', 17501, 10),
                                 ('keyword_list', 14256, 10), ('tag_list', 330, 10)])
    parser.add_argument('--freeze_multi_modal_emb', type=bool, default=True)
    parser.add_argument('--multi_modal_emb_matrix', type=np.ndarray)
    parser.add_argument('--multi_modal_emb_matrix_path', type=str,
                        default=os.path.join(W2V_PATH, 'embedding.npy'))
    parser.add_argument('--multi_modal_emb_char_matrix', type=np.ndarray)
    parser.add_argument('--multi_modal_emb_char_matrix_path', type=str,
                        default=os.path.join(W2V_CHAR_PATH, 'embedding.npy'))
    parser.add_argument('--multi_modal_emb_size', type=int, default=200)
    parser.add_argument('--merge_data_path', type=str, default=os.path.join(TRAIN_TEST_DATA_PATH, 'nn_train.csv'))
    parser.add_argument('--test_data_path', type=str, default=os.path.join(TRAIN_TEST_DATA_PATH, 'nn_test.csv'))
    parser.add_argument('--submit_test_path', type=str, default=os.path.join(COMPETITION_DATA_PATH, 'test_b.csv'))
    parser.add_argument('--round1_user_action_path', type=str, default=os.path.join(COMPETITION_DATA_PATH0, 'user_action.csv'))
    parser.add_argument('--round2_user_action_path', type=str, default=os.path.join(COMPETITION_DATA_PATH, 'user_action.csv'))
    parser.add_argument('--user_token2id_path', type=str, default=os.path.join(UID_FID_SVD_PATH, 'user_token2id.json'))
    parser.add_argument('--uid_fid_vector_path', type=str, default=os.path.join(UID_FID_SVD_PATH,'uid_svd_64.npy'))
    parser.add_argument('--feed_token2id_path', type=str, default=os.path.join(UID_FID_SVD_PATH,'feed_token2id.json'))
    parser.add_argument('--fid_vector_path', type=str, default=os.path.join(UID_FID_SVD_PATH,'fid_svd_64.npy'))
    parser.add_argument('--author_token2id_path', type=str,
                        default=os.path.join(UID_AID_SVD_PATH,'author_token2id.json'))
    parser.add_argument('--uid_aid_vector_path', type=str, default=os.path.join(UID_AID_SVD_PATH,'uid_svd_16.npy'))
    parser.add_argument('--aid_vector_path', type=str, default=os.path.join(UID_AID_SVD_PATH, 'aid_svd_16.npy'))
    parser.add_argument('--bgm_song_token2id_path', type=str,
                        default=os.path.join(UID_BGM_SONG_ID_SVD_PATH, 'bgm_song_token2id.json'))
    parser.add_argument('--uid_bgm_song_id_vector_path', type=str,
                        default=os.path.join(UID_BGM_SONG_ID_SVD_PATH, 'uid_svd_16.npy'))
    parser.add_argument('--bgm_song_id_vector_path', type=str,
                        default=os.path.join(UID_BGM_SONG_ID_SVD_PATH, 'bgm_song_id_svd_16.npy'))
    parser.add_argument('--bgm_singer_token2id_path', type=str,
                        default=os.path.join(UID_BGM_SINGER_ID_SVD_PATH,  'bgm_singer_token2id.json'))
    parser.add_argument('--uid_bgm_singer_id_vector_path', type=str,
                        default=os.path.join(UID_BGM_SINGER_ID_SVD_PATH, 'uid_svd_16.npy'))
    parser.add_argument('--bgm_singer_id_vector_path', type=str,
                        default=os.path.join(UID_BGM_SINGER_ID_SVD_PATH, 'bgm_singer_id_svd_16.npy'))
    parser.add_argument('--keyword_list_token2id_path', type=str,
                        default=os.path.join(UID_KEYWORD_LIST_SVD_PATH,  'keyword_list_token2id.json'))
    parser.add_argument('--uid_keyword_list_vector_path', type=str,
                        default=os.path.join(UID_KEYWORD_LIST_SVD_PATH, 'uid_svd_16.npy'))
    parser.add_argument('--keyword_list_vector_path', type=str,
                        default=os.path.join(UID_KEYWORD_LIST_SVD_PATH, 'keyword_list_svd_16.npy'))
    parser.add_argument('--tag_list_token2id_path', type=str,
                        default=os.path.join(UID_TAG_LIST_SVD_PATH,  'tag_list_token2id.json'))
    parser.add_argument('--uid_tag_list_vector_path', type=str,
                        default=os.path.join(UID_TAG_LIST_SVD_PATH, 'uid_svd_16.npy'))
    parser.add_argument('--tag_list_vector_path', type=str,
                        default=os.path.join(UID_TAG_LIST_SVD_PATH, 'tag_list_svd_16.npy'))

    # deepwalk
    parser.add_argument('--userid_feedid_userid_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH, 'userid_feedid_userid_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_feedid_feedid_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH, 'userid_feedid_feedid_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_authorid_userid_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH,'userid_authorid_userid_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_authorid_authorid_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH,'userid_authorid_authorid_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_bgm_song_id_bgm_song_id_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH,'userid_bgm_song_id_bgm_song_id_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_bgm_song_id_userid_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH, 'userid_bgm_song_id_userid_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_bgm_singer_id_bgm_singer_id_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH, 'userid_bgm_singer_id_bgm_singer_id_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_bgm_singer_id_userid_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH, 'userid_bgm_singer_id_userid_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_keyword_list_userid_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH, 'userid_keyword_list_userid_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_keyword_list_keyword_list_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH, 'userid_keyword_list_keyword_list_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_tag_list_userid_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH, 'userid_tag_list_userid_dev_deepwalk_64.pkl'))
    parser.add_argument('--userid_tag_list_tag_list_deepwalk', type=str,
                        default=os.path.join(DEEPWALK_PATH, 'userid_tag_list_tag_list_dev_deepwalk_64.pkl'))

                        
    parser.add_argument('--vocab_path', type=str, default=os.path.join(W2V_PATH, 'token2id.json'))
    parser.add_argument('--vocab_size', type=int, default=70000)
    parser.add_argument('--char_vocab_path', type=str, default=os.path.join(W2V_CHAR_PATH, 'token2id.json'))
    parser.add_argument('--char_vocab_size', type=int, default=20500)
    parser.add_argument('--keyword_vocab_path', type=str, default=os.path.join(UID_KEYWORD_LIST_SVD_PATH, 'keyword_list_token2id.json'))
    parser.add_argument('--keyword_vocab_size', type=int, default=14256)
    parser.add_argument('--tag_vocab_path', type=str, default=os.path.join(UID_TAG_LIST_SVD_PATH, 'tag_list_token2id.json'))
    parser.add_argument('--tag_vocab_size', type=int, default=330)

    parser.add_argument('--feed_embedding_path', type=str, default=os.path.join(COMPETITION_DATA_PATH, 'feed_embeddings.csv'))

    parser.add_argument('--num_folds', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--bs', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default="")
    parser.add_argument('--ema_start', type=bool, default=False)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--ema_start_step', type=int, default=40)
    parser.add_argument('--bucket_size_multiplier', type=int, default=10)
    parser.add_argument('--best_model_path', type=str, default='')
    parser.add_argument('--debug_data', action='store_true')

    args = parser.parse_args()
    args.output_dir = os.path.join(MODEL_PATH, "nn_" + str(args.seed))
    return args
