import os
import pickle
import gzip
from pref_db import PrefDB, PrefBuffer


def merge_pickles(file_list, output_file, maxlen):
    merged_prefdb = PrefDB(maxlen)
    i = 0

    for file in file_list:
        prefdb = PrefDB.load(file)
        for k1, k2, pref in prefdb.prefs:
            s1 = prefdb.segments[k1]
            s2 = prefdb.segments[k2]
            merged_prefdb.append(s1, s2, pref)
            i+=1
            print(i)

    merged_prefdb.save(output_file)

# マージしたデータを保存するファイル名
output_file_ = 'runs/merged/train.pkl.gz'
output_file = 'runs/merged/val.pkl.gz'


# マージしたいファイルのリスト
file_list_ = [ 
             'runs/1688422400_e2f5ab2/train.pkl.gz',
             'runs/1688428361_e2f5ab2/train.pkl.gz',
             'runs/1688595796_e2f5ab2/train.pkl.gz',
             'runs/1688600764_e2f5ab2/train.pkl.gz',
             'runs/1688609248_e2f5ab2/train.pkl.gz',
             'runs/1688620732_e2f5ab2/train.pkl.gz',
             'runs/1688773066_af61ff4/train.pkl.gz',
             'runs/1689205083_af61ff4/train.pkl.gz',
             'runs/1689279345_af61ff4/train.pkl.gz', 
             'runs/1689284873_af61ff4/train.pkl.gz']  # ここにファイル名を追加してください


# マージしたいファイルのリスト
file_list = [ 
             'runs/1688422400_e2f5ab2/val.pkl.gz',
             'runs/1688428361_e2f5ab2/val.pkl.gz',
             'runs/1688595796_e2f5ab2/val.pkl.gz',
             'runs/1688600764_e2f5ab2/val.pkl.gz',
             'runs/1688609248_e2f5ab2/val.pkl.gz',
             'runs/1688620732_e2f5ab2/val.pkl.gz',
             'runs/1688773066_af61ff4/val.pkl.gz',
             'runs/1689205083_af61ff4/val.pkl.gz',
             'runs/1689279345_af61ff4/val.pkl.gz', 
             'runs/1689284873_af61ff4/val.pkl.gz']  # ここにファイル名を追加してください

maxlen = 1250  # 適切な値に変更してください

merge_pickles(file_list, output_file, maxlen)