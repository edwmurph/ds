import pandas as pd
import statsmodels.api as sm

def processSubset(feature_set, X, y):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def getBest(k, X, y):

    tic = time.time()

    results = []

    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo, X, y))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].idxmin()]

    toc = time.time()
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc-tic), "seconds.")

    # Return the best model, along with some other useful information about the model
    return best_model

def getAllBest(X, y):
    models_best = pd.DataFrame(columns=["RSS", "model"])

    tic = time.time()
    for i in range(1,14):
        models_best.loc[i] = getBest(i, X, y)

    toc = time.time()
    print("Total elapsed time:", (toc-tic), "seconds.")
    return models_best
