

import os
from pref_db import PrefDB, PrefBuffer
from os import path as osp


prefs_dir = "/media/user/805f81f1-eebc-4f25-942f-bba29fb6c676/berkeley_models/rlhf_human_batch1"
train_path = osp.join(prefs_dir, 'train_3k.pkl.gz')
pref_db_train = PrefDB.load(train_path)
print("Loaded training preferences from '{}'".format(train_path))
n_prefs, n_segs = len(pref_db_train), len(pref_db_train.segments)
print("({} preferences, {} segments)".format(n_prefs, n_segs))
