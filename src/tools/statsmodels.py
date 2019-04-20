import statsmodels.api as sm
from statsmodels.graphics.gofplots import ProbPlot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time
import itertools


#
# My collection of helper functions that work with statsmodels library
#


""" Build model
import statsmodels.formula.api as smf
import statsmodels.api as sm
load_datasets('Smarket.csv')

(
    formula = 'Sales ~ Price'
    df      = pd.read_csv('../data/Carseats.csv', index_col=0)
    model   = smf.ols(formula, df).fit()
or
    Smarket = pd.read_csv('../data/Smarket.csv', index_col=0)
    formula = 'Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume'
    model   = smf.glm(formula, Smarket, family=sm.families.Binomial()).fit()
)

model.summary()
"""


def regression_plots_2d(model, df, x, y, f):
    """ Visualize fit of 2d models and their regression plots

    model = result of fitting a statsmodels.api model
    df    = dataframe
    x     = the name of the predictor
    y     = the name of the dependent variable
    f     = function to apply coefficients to. E.g.: lambda x,a,b: a + b*x
    """
    fig = plt.figure(figsize=[12, 5])
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # PLOT 1: add observations as scatter plot
    ax1.scatter(df[x], df[y], s=7, label=None)

    # PLOT 1: add regression line
    X1 = np.linspace(df[x].min(), df[x].max())
    Y1 = f(X1, *model.params.values)
    ax1.plot(X1, Y1, 'red', label=None)

    ax1.set_title('Regressing {} onto {}'.format(y, x))
    ax1.set_xlabel(x)
    ax1.set_ylabel(y)
    yRange = df[y].max() - df[y].min()
    ax1.set_ylim(df[y].min() - yRange/10, df[y].max() + yRange/10)
    ax1.grid()

    # PLOT 2: add residuals as scatter plot
    X2 = model.fittedvalues
    Y2 = model.resid
    ax2.scatter(X2, Y2, s=10)
    # PLOT 2: add smoother
    smoother = sm.nonparametric.lowess(Y2, X2, frac=.3)
    ax2.plot(smoother[:, 0], smoother[:, 1], 'red', label='smoother')
    ax2.set_title('Residual plot')
    ax2.set_xlabel('fitted values')
    ax2.set_ylabel('residuals')
    ax2.legend()
    ax2.grid()

    plt.show()


def diagnostic_plots(model, df):
    """ Reproduces the 4 base plots of an OLS model in R.

    model = a statsmodel.api.OLS model after regression
    df    = dataframe used in regression
    """
    # normalized residuals
    model_norm_residuals = model.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model.resid)
    # leverage, from statsmodels internals
    model_leverage = model.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model.get_influence().cooks_distance[0]

    fig = plt.figure(figsize=[12, 10])
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    plt.subplots_adjust(hspace=.3, wspace=.2)

    X = model.fittedvalues
    Y = model.resid
    ax1.scatter(X, Y, s=10)
    smoother = sm.nonparametric.lowess(Y, X, frac=.3)
    ax1.plot(smoother[:, 0], smoother[:, 1], 'red')
    ax1.set_title('Residuals vs Fitted')
    ax1.set_xlabel('Fitted values')
    ax1.set_ylabel('Residuals')
    ax1.grid()
    # annotations
    abs_resid = model_abs_resid.sort_values(ascending=False)
    abs_resid_top_3 = abs_resid[:3]
    for i in abs_resid_top_3.index:
        ax1.annotate(i,
                     xy=(model.fittedvalues[i],
                         model.resid[i]))

    QQ = ProbPlot(model_norm_residuals)
    QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1, ax=ax2)
    ax2.set_title('Normal Q-Q')
    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Standardized Residuals')
    ax2.grid()
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for r, i in enumerate(abs_norm_resid_top_3):
        ax2.annotate(i,
                     xy=(np.flip(QQ.theoretical_quantiles, 0)[r],
                         model_norm_residuals[i]))

    ax3.scatter(model.fittedvalues, model_norm_residuals_abs_sqrt, alpha=0.5)
    sns.regplot(model.fittedvalues, model_norm_residuals_abs_sqrt,
                scatter=False,
                ax=ax3,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    ax3.set_title('Scale-Location')
    ax3.set_xlabel('Fitted values')
    ax3.set_ylabel(r'$\sqrt{|Standardized Residuals|}$')
    ax3.grid()
    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
    for i in abs_norm_resid_top_3:
        ax3.annotate(i,
                     xy=(model.fittedvalues[i],
                         model_norm_residuals_abs_sqrt[i]))

    ax4.scatter(model_leverage, model_norm_residuals, alpha=0.5)
    sns.regplot(model_leverage, model_norm_residuals,
                ax=ax4,
                scatter=False,
                ci=False,
                lowess=True,
                line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    ax4.set_xlim(0, max(model_leverage)+0.01)
    ax4.set_ylim(-3, 5)
    ax4.set_title('Residuals vs Leverage')
    ax4.set_xlabel('Leverage')
    ax4.set_ylabel('Standardized Residuals')
    ax4.grid()
    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
    for i in leverage_top_3:
        ax4.annotate(i,
                     xy=(model_leverage[i],
                         model_norm_residuals[i]))

    p = len(model.params)  # number of model parameters

    def f(x): return np.sqrt((0.5 * p * (1 - x)) / x)
    X = np.linspace(0.001, max(model_leverage), 50)
    Y = f(X)
    ax4.plot(X, Y, label='Cook\'s distance', lw=1, ls='--', color='red')

    def f(x): return np.sqrt((1 * p * (1 - x)) / x)
    X = np.linspace(0.001, 0.200, 50)
    Y = f(X)
    ax4.plot(X, Y, lw=1, ls='--', color='red')
    ax4.legend()


def processSubset(feature_set, X, y):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y, X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model": regr, "RSS": RSS}


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
    print(
        "Processed",
        models.shape[0],
        "models on",
        k,
        "predictors in",
        (toc-tic),
        "seconds.")

    # Return the best model, along with some other useful information about
    # the model
    return best_model


def best_subset_selection(X, y):
    models_best = pd.DataFrame(columns=["RSS", "model"])

    tic = time.time()
    for i in range(1, 14):
        models_best.loc[i] = getBest(i, X, y)

    toc = time.time()
    print("Total elapsed time:", (toc-tic), "seconds.")
    return models_best
