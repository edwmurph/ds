

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


''' TODO automate removing column with more missing values '''


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
