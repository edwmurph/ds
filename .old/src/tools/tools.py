import os
import subprocess
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score

import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz

pd.set_option('display.max_columns', None)


# https://github.com/MTG/sms-tools/issues/36
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")



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
# Data preprocessing
#


def drop_cols_with_only_val_when_grouped(df, groupby, vals, threshold):
    grouped = df.groupby(groupby).aggregate(
        lambda x: all(any(item == val for val in vals) for item in x.values))
    cols = []
    for column in grouped.columns:
        series = grouped[column].value_counts()/grouped[column].count()
        if True in series.index:
            if series[True] > threshold:
                cols.append(column)
    return df.drop(columns=cols, axis=1)



def del_cols_with_1_val(df):
    columns = []
    for column in df.columns:
        if len(df[column].value_counts()) == 1:
            columns.append(column)
    return df.drop(columns, axis=1)

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


def label_encoder(df, columns):
    le_dict = {}
    #df_quan = df.drop(qualatatives, axis=1)
    for c in columns:
        le = LabelEncoder()
        le.fit(df[c].unique())
        df[c] = le.transform(df[c])
        le_dict[c] = le
    return df, le_dict


def normalize(df, exclusions=[]):
    result = df.copy()
    for feature_name in [x for x in df.columns if x not in exclusions]:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (
            df[feature_name] - min_value) / (max_value - min_value)
    return result


def print_tree(model, features):
    dot_data = export_graphviz(model, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return Image(graph.create_png())
