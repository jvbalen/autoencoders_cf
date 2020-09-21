"""
From https://github.com/younggyoseo/vae-cf-pytorch with minor changes
"""
import os
import sys

import pandas as pd
from scipy import sparse
import numpy as np


class DataLoader():
    '''
    Load Movielens-20m dataset
    '''
    def __init__(self, path):
        self.pro_dir = os.path.join(path, 'pro_sg')
        assert os.path.exists(self.pro_dir), "Preprocessed files does not exist. Run data.py"

        self.n_items = self.load_n_items()

    def load_data(self, split='train'):
        if split == 'train':
            return self._load_train_data()
        elif split == 'validation':
            return self._load_tr_te_data(split)
        elif split == 'test':
            return self._load_tr_te_data(split)
        else:
            raise ValueError("datatype should be in [train, validation, test]")

    def load_n_items(self):
        unique_sid = list()
        with open(os.path.join(self.pro_dir, 'unique_sid.txt'), 'r') as f:
            for line in f:
                unique_sid.append(line.strip())
        n_items = len(unique_sid)
        return n_items

    def _load_train_data(self):
        """Return x_train, x_train.copy() as there is no y_train
        """
        path = os.path.join(self.pro_dir, 'train.csv')

        tp = pd.read_csv(path)
        n_users = tp['uid'].max() + 1

        rows, cols = tp['uid'], tp['sid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items))
        return data, None

    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['sid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['sid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64',
                                    shape=(end_idx - start_idx + 1, self.n_items))
        return data_tr, data_te


def get_count(tp, id):
    playcount_groupbyid = tp[[id]].groupby(id, as_index=True)
    count = playcount_groupbyid.size()
    return count


def filter_triplets(tp, min_uc=5, min_sc=0):
    if min_sc > 0:
        item_count = get_count(tp, item_col)
        items_to_keep = item_count.index[item_count >= min_uc]
        if len(items_to_keep) < len(item_count):
            print(f'Dropping {len(item_count) - len(items_to_keep)} of {len(item_count)} items...')
            tp = tp[tp[item_col].isin(items_to_keep)]

    if min_uc > 0:
        user_count = get_count(tp, user_col)
        users_to_keep = user_count.index[user_count >= min_uc]
        if len(users_to_keep) < len(user_count):
            print(f'Dropping {len(user_count) - len(users_to_keep)} of {len(user_count)} users...')
            tp = tp[tp[user_col].isin(users_to_keep)]

    user_count, item_count = get_count(tp, user_col), get_count(tp, item_col)
    return tp, user_count, item_count


def split_train_test_proportion(data, test_prop=0.2):
    data_grouped_by_user = data.groupby(user_col)
    tr_list, te_list = list(), list()

    np.random.seed(98765)

    for _, group in data_grouped_by_user:
        n_items_u = len(group)

        if n_items_u >= 5:
            idx = np.zeros(n_items_u, dtype='bool')
            sample = np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False)
            idx[sample.astype('int64')] = True

            tr_list.append(group[np.logical_not(idx)])
            te_list.append(group[idx])

        else:
            tr_list.append(group)

    data_tr = pd.concat(tr_list)
    data_te = pd.concat(te_list)

    return data_tr, data_te


def numerize(tp, profile2id, show2id):
    uid = tp[user_col].apply(lambda x: profile2id[x])
    sid = tp[item_col].apply(lambda x: show2id[x])
    return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])


if __name__ == '__main__':

    path = sys.argv[1] if len(sys.argv) > 1 else 'ml-20m/ratings.csv'
    delimiter = sys.argv[2] if len(sys.argv) > 2 else None
    has_header = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    data_dir = os.path.dirname(path)

    MIN_RATING = 3.5
    MIN_USER_COUNT = 5
    N_HELDOUT_USERS = 10000
    RANDOM_SEED = 98765

    # Load Data
    print("Load and preprocess dataset")
    raw_data = pd.read_csv(path, header=0 if has_header else None, delimiter=delimiter)
    if len(raw_data.columns) > 2:
        user_col, item_col, rating_col = raw_data.columns[:3]
        raw_data = raw_data[raw_data[rating_col] > MIN_RATING]
    else:
        user_col, item_col = raw_data.columns

    # Filter Data
    raw_data, user_activity, item_popularity = filter_triplets(raw_data, min_uc=MIN_USER_COUNT)

    # Shuffle User Indices
    unique_uid = user_activity.index
    np.random.seed(RANDOM_SEED)
    idx_perm = np.random.permutation(unique_uid.size)
    unique_uid = unique_uid[idx_perm]

    n_users = unique_uid.size

    # Split Train/Validation/Test User Indices
    tr_users = unique_uid[:(n_users - N_HELDOUT_USERS * 2)]
    vd_users = unique_uid[(n_users - N_HELDOUT_USERS * 2): (n_users - N_HELDOUT_USERS)]
    te_users = unique_uid[(n_users - N_HELDOUT_USERS):]

    train_plays = raw_data.loc[raw_data[user_col].isin(tr_users)]
    unique_sid = pd.unique(train_plays[item_col])

    show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
    profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

    pro_dir = os.path.join(data_dir, 'pro_sg')

    if not os.path.exists(pro_dir):
        os.makedirs(pro_dir)

    with open(os.path.join(pro_dir, 'unique_sid.txt'), 'w') as f:
        for sid in unique_sid:
            f.write('%s\n' % sid)

    vad_plays = raw_data.loc[raw_data[user_col].isin(vd_users)]
    vad_plays = vad_plays.loc[vad_plays[item_col].isin(unique_sid)]

    vad_plays_tr, vad_plays_te = split_train_test_proportion(vad_plays)

    test_plays = raw_data.loc[raw_data[user_col].isin(te_users)]
    test_plays = test_plays.loc[test_plays[item_col].isin(unique_sid)]

    test_plays_tr, test_plays_te = split_train_test_proportion(test_plays)

    train_data = numerize(train_plays, profile2id, show2id)
    train_data.to_csv(os.path.join(pro_dir, 'train.csv'), index=False)

    vad_data_tr = numerize(vad_plays_tr, profile2id, show2id)
    vad_data_tr.to_csv(os.path.join(pro_dir, 'validation_tr.csv'), index=False)

    vad_data_te = numerize(vad_plays_te, profile2id, show2id)
    vad_data_te.to_csv(os.path.join(pro_dir, 'validation_te.csv'), index=False)

    test_data_tr = numerize(test_plays_tr, profile2id, show2id)
    test_data_tr.to_csv(os.path.join(pro_dir, 'test_tr.csv'), index=False)

    test_data_te = numerize(test_plays_te, profile2id, show2id)
    test_data_te.to_csv(os.path.join(pro_dir, 'test_te.csv'), index=False)

    print("Done!")
