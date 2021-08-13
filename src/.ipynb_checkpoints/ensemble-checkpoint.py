import pandas as pd
from .common_path import *

y_list = ["read_comment", "like", "click_avatar", "forward", 'favorite', 'comment', 'follow']

nn = []

for path in os.listdir(SUBMISSION_PATH):
    if 'nn' in path:
        nn.append(pd.read_csv(os.path.join(SUBMISSION_PATH, path)))
        
ensemble_df = nn[0][['userid', 'feedid']].copy()

nn_ensemble = sum(sub[y_list] for sub in nn) / len(nn)

ensemble_df.loc[:, y_list] = nn_ensemble

ensemble_df.to_csv(os.path.join(SUBMISSION_PATH, 'result.csv'), index=False, encoding='utf8')
