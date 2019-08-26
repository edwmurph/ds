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


def load_python_data(ds, df_dict=False, index_col=False, encoding='utf-8'):
    if df_dict:
        ''' eg. df_dict = {'actual name': { 'dtype': object, 'name': 'new' }'''
        dtype = {k: v['dtype'] for k, v in df_dict.items()}
        cols = {k: v['name'] or k for k, v in df_dict.items()}
        df = pd.read_csv(
            '../data/{}.csv'.format(ds),
            usecols=df_dict,
            dtype=dtype,
            encoding=encoding
        )
        if index_col:
            df = df.set_index(index_col)
        df = trim_str_vals(df)
        df = df.replace(r'^\s*$', np.nan, regex=True)
        return df.rename(index=str, columns=cols)
    else:
        df = pd.read_csv('../data/{}.csv'.format(ds), encoding=encoding)
        df = trim_str_vals(df)
        df = df.replace(r'^\s*$', np.nan, regex=True)
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


def trim_str_vals(df):
    df_str = df.select_dtypes(['object'])
    df[df_str.columns] = df_str.apply(lambda x: x.str.strip())
    return df


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


def build_clf(features, categoricals, model):
    numeric_features = [i for i in features if i not in categoricals]
    categorical_features = [i for i in features if i in categoricals]
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[('onehot', OneHotEncoder(categories='auto'))])

    transformers = []

    if len(numeric_features) > 0:
        transformers = transformers + [('num',
                                        numeric_transformer,
                                        numeric_features)]

    if len(categorical_features) > 0:
        transformers = transformers + [('cat',
                                        categorical_transformer,
                                        categorical_features)]

    preprocessor = ColumnTransformer(transformers)

    return Pipeline(
        steps=[('preprocessor', preprocessor),
               ('classifier', model)])


def processSubset(features, X, y, model, categoricals):
    clf = build_clf(features, categoricals, model)
    score = cross_val_score(clf, X[features], y, cv=5).mean()
    return {'feature_set': features,
            'num_features': len(features),
            'score': score}


def getBest(k, X, y, model, quiet, categoricals):
    tic = time.time()
    results = []
    durations = []

    for combo in itertools.combinations(X.columns, k):
        start = time.time()
        result = processSubset(list(combo), X, y, model, categoricals)
        stop = time.time()
        durations.append(stop - start)
        results.append(result)

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    best_model = models.loc[models['score'].idxmax()]

    toc = time.time()
    seconds = (toc-tic)

    if not quiet:
        print(
            time.asctime(),
            "Processed",
            str(int(models.shape[0])).rjust(10),
            "models on",
            str(int(k)).rjust(2),
            "features in",
            str(int(seconds / (60*60))).rjust(2) + 'h',
            str(int(seconds / 60 % 60)).rjust(2) + 'm',
            str(int(seconds % 60)).rjust(2) + 's',
            '|',
            'avg ' + '%.5f' % np.mean(durations) + 's/model',
            '|',
            '%.5f' % best_model.score,
            )

    return best_model


def best_subset_selection(
        X, y, model, quiet=False, categoricals=[], lowerLim=False,
        upperLim=False, tol=3):
    best_models = pd.DataFrame(
        columns=[
            'score',
            'feature_set',
            'num_features'])

    tic = time.time()
    lower = lowerLim or 1
    upper = upperLim or len(X.columns) + 1
    for i in range(lower, upper):
        best_models.loc[i] = getBest(i, X, y, model, quiet, categoricals)
        range_start = i-tol-1
        if best_models.index.contains(range_start):
            score_range_start = best_models.loc[range_start].score
            cont = False
            for k in [j for j in range(range_start + 1, i+1)]:
                score = best_models.loc[k].score
                if score > score_range_start:
                    cont = True
            if cont:
                continue
            else:
                break

    toc = time.time()
    seconds = (toc-tic)
    print("Total elapsed time:",
          str(int(seconds / (60*60))).rjust(2) + 'h',
          str(int(seconds / 60 % 60)).rjust(2) + 'm',
          str(int(seconds % 60)).rjust(2) + 's'
          )

    if not quiet:
        print('\nBest models for each number of features:')
        print(best_models)

    best_num_features = best_models.score.idxmax()
    best_score = best_models.score.max()
    worst_score = best_models.score.min()
    best_features = best_models[best_models.score ==
                                best_score].feature_set.values[0]

    if not quiet:
        print('\nBest features:\n', best_features)
        print('\nBest features score:', best_score, 'worst:', worst_score)
        plt.plot(best_models.num_features, best_models.score)
        plt.annotate(
            '%0.5f' % best_score, xy=(best_num_features, best_score),
            xytext=(best_num_features, (best_score + worst_score) / 2),
            arrowprops=dict(arrowstyle="->"))
        plt.title('Best subset selection')
        plt.xlabel('num features')
        plt.ylabel('mean accuracy score')

    return best_score, best_features


def print_tree(model, features):
    dot_data = export_graphviz(model, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data)
    return Image(graph.create_png())
