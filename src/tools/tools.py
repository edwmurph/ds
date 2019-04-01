# https://github.com/MTG/sms-tools/issues/36
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
import os
import subprocess
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


""" Call this at start of scripts that depend on having  """
def load_datasets(dataset):
    exists = os.path.isfile('../data/{}'.format(dataset))
    if exists:
        print('data already loaded')
    else:
        subprocess.call(["/usr/local/bin/Rscript", "--vanilla", "../scripts/loadData.r"])
        existsAfter = os.path.isfile('../data/{}'.format(dataset))
        if existsAfter:
            print('downloaded data')
        else:
            print('Requested dataset not downloaded. Check scripts/loadData.r R script')


#
# Matplotlib
#

""" Create figure with multiple subplots

fig = plt.figure(figsize=[12,10])
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
plt.subplots_adjust(hspace=.3, wspace=.2)
"""

""" Graph line

f = lambda x: np.sqrt(x)
X = np.linspace(0, 10)
Y = f(X)
ax1.plot(X, Y, label='sqrt fn', lw=1, ls='--', color='red')
"""


#
# General data structure operations
#

""" replace_in_series(df.column, r'\d*-([a-zA-Z]*)-\d*', '\\1') """
def replace_in_series(series, regex, replacement):
    return series.apply(
        lambda x: re.sub(regex, replacement, x)
    )

def validate_lists_equal(list1, list2):
    arr1 = np.sort( np.array(list1) )
    arr2 = np.sort( np.array(list2) )
    if not np.array_equal( arr1, arr2 ):
        raise Exception('lists are not equal:')

def map_dic(dic, valFn, keyFn = lambda x,y: x):
    newDic = {}
    for key, value in dic.items():
        newKey = keyFn(key, value)
        newValue = valFn(key, value)
        newDic[ newKey ] = newValue
    return newDic

#
# Data analysis
#

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, threshold = 0.8):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[au_corr>threshold]
