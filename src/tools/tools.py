import os
import subprocess
import re
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)


# https://github.com/MTG/sms-tools/issues/36
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


def load_python_data(ds, df_dict):
    ''' e.g. df_dict = {'actual name': { 'dtype': object, 'name': 'new' }'''
    dtype = {k: v['dtype'] for k, v in df_dict.items()}
    cols = {k: v['name'] for k, v in df_dict.items()}
    df = pd.read_csv(
        '../data/{}.csv'.format(ds),
        usecols=df_dict,
        dtype=dtype
    )
    df = df.rename(index=str, columns=cols)
    return df


def load_r_data(ds):
    ''' Call this at start of scripts that depend on having an R dataset '''
    pathToData = '../data/{}.csv'.format(ds)
    exists = os.path.isfile(pathToData)
    if exists:
        return pd.read_csv(pathToData, index_col=0)
    else:
        args = ["/usr/local/bin/Rscript", "--vanilla", "../scripts/loadData.r"]
        subprocess.call(args)
        exists_after = os.path.isfile(pathToData)
        if exists_after:
            return pd.read_csv(pathToData, index_col=0)
        else:
            print('Dataset not downloaded. Check scripts/loadData.r R script')


#
# Matplotlib
#

''' Create figure with multiple subplots

fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
plt.subplots_adjust(hspace=.3, wspace=.2)
'''

''' Graph line

f = lambda x: np.sqrt(x)
X = np.linspace(0, 10)
Y = f(X)
ax1.plot(X, Y, label='sqrt fn', lw=1, ls='--', color='red')
'''


#
# General data structure operations
#


def replace_in_series(series, regex, replacement):
    ''' replace_in_series(df.column, '\\d*-([a-zA-Z]*)-\\d*', '\\1') '''
    return series.apply(
        lambda x: re.sub(regex, replacement, x)
    )


def validate_lists_equal(list1, list2):
    arr1 = np.sort(np.array(list1))
    arr2 = np.sort(np.array(list2))
    if not np.array_equal(arr1, arr2):
        raise Exception('lists are not equal:')


def map_dic(dic, valFn, keyFn=lambda x, y: x):
    newDic = {}
    for key, value in dic.items():
        newKey = keyFn(key, value)
        newValue = valFn(key, value)
        newDic[newKey] = newValue
    return newDic

#
# Data analysis
#


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (
            df[feature_name] - min_value) / (max_value - min_value)
    return result


def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def get_top_abs_correlations(df, threshold=0.8):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[au_corr > threshold]
