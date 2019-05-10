import os
import subprocess
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import itertools

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict, cross_val_score

import pydotplus
from IPython.display import Image
from sklearn.tree import export_graphviz

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


def label_encoder(df, columns):
    le_dict = {}
    #df_quan = df.drop(qualatatives, axis=1)
    for c in columns:
        le = LabelEncoder()
        le.fit(df[c].unique())
        df[c] = le.transform(df[c])
        le_dict[c] = le
    return df, le_dict


def min_max_scaler(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


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


def get_top_abs_correlations(df, threshold=0.8, num_corr=7):
    corr_df = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    corr_df = corr_df.drop(labels=labels_to_drop)
    corr_df = corr_df.reindex(corr_df.abs().sort_values(ascending=False).index)
    corr_df = corr_df[corr_df.abs() > threshold]
    correlated = set()
    i = 0
    for pair in corr_df.index.values:
        for feature in pair:
            if i < num_corr:
                correlated.add(feature)
                i = i + 1

    return corr_df, list(correlated)


def processSubset(feature_set, X, y, model, numerical):
    if numerical:
        pred = cross_val_predict(model, X[feature_set], y, cv=5)
        mse = mean_squared_error(y, pred)
        return {'feature_set': feature_set,
                'num_features': len(feature_set), 'mse': mse}
    else:
        score = cross_val_score(model, X[feature_set], y, cv=5).mean()
        return {'feature_set': feature_set,
                'num_features': len(feature_set), 'score': score}


def getBest(k, X, y, model, quiet, numerical):
    tic = time.time()
    results = []

    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(list(combo), X, y, model, numerical))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    if numerical:
        best_model = models.loc[models['mse'].idxmin()]
    else:
        best_model = models.loc[models['score'].idxmax()]

    toc = time.time()
    if not quiet:
        print(
            "Processed",
            models.shape[0],
            "models on",
            k,
            "features in",
            (toc-tic),
            "seconds.")

    # Return the best model, along with some other useful information about
    # the model
    return best_model


def best_subset_selection(X, y, model, quiet=False, numerical=True):
    num_features = len(X.columns) + 1
    if numerical:
        best_models = pd.DataFrame(
            columns=[
                'mse',
                'feature_set',
                'num_features'])
    else:
        best_models = pd.DataFrame(
            columns=[
                'score',
                'feature_set',
                'num_features'])

    tic = time.time()
    for i in range(1, num_features):
        best_models.loc[i] = getBest(i, X, y, model, quiet, numerical)

    toc = time.time()
    print("Total elapsed time:", (toc-tic), "seconds.")
    if not quiet:
        print('\nBest models for each number of features:')
        print(best_models)

    if numerical:
        best_num_features = best_models.mse.idxmin()
        best_mse = best_models.mse.min()
        worst_mse = best_models.mse.max()
        best_features = best_models[best_models.mse ==
                                    best_mse].feature_set.values[0]
    else:
        best_num_features = best_models.score.idxmax()
        best_score = best_models.score.max()
        worst_score = best_models.score.min()
        best_features = best_models[best_models.score ==
                                    best_score].feature_set.values[0]

    if not quiet:
        print('\nBest features:\n', best_features)
        if numerical:
            print('\nBest features mse:', best_mse, 'worst:', worst_mse)
            plt.plot(best_models.num_features, best_models.mse)
            plt.annotate(
                '%0.5f' % best_mse, xy=(best_num_features, best_mse),
                xytext=(best_num_features, (best_mse + worst_mse) / 2),
                arrowprops=dict(arrowstyle="->"))
            plt.title('Best subset selection')
            plt.xlabel('num features')
            plt.ylabel('MSE')
        else:
            print('\nBest features score:', best_score, 'worst:', worst_score)
            plt.plot(best_models.num_features, best_models.score)
            plt.annotate(
                '%0.5f' % best_score, xy=(best_num_features, best_score),
                xytext=(best_num_features, (best_score + worst_score) / 2),
                arrowprops=dict(arrowstyle="->"))
            plt.title('Best subset selection')
            plt.xlabel('num features')
            plt.ylabel('mean accuracy score')
    if numerical:
        return best_mse, best_features
    else:
        return best_score, best_features


def print_tree(model, features):
    dot_data = export_graphviz(model, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return Image(graph.create_png())
