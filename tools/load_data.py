import pandas as pd


def trim_str_vals(df):
    df_str = df.select_dtypes(['object'])
    df[df_str.columns] = df_str.apply(lambda x: x.str.strip())
    return df


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
