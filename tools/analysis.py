import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score


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
