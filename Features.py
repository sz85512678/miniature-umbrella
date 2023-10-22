import numpy as np
from pandas import CategoricalDtype


def make_features_naive(df_in):
    df = df_in.copy()
    df.pop('Product ID')
    df.pop('UDI')
    df['Type'] = df['Type'].astype(CategoricalDtype(['L', 'M', 'H'], ordered=True))
    df['Type'] = df['Type'].cat.codes

    y1 = df.pop('Target')
    y2 = df.pop('Failure_Type')
    return df, y1, y2


def make_features(df_in):
    df = df_in.copy()
    df.pop('Product ID')
    df.pop('UDI')
    df['Type'] = df['Type'].astype(CategoricalDtype(['L', 'M', 'H'], ordered=True))
    df['Type'] = df['Type'].cat.codes

    y1 = df.pop('Target')
    y2 = df.pop('Failure_Type')
    df = make_maths_transforms(df, df.columns)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(df.median())
    return df, y1, y2


def make_maths_transforms(data, features):
    n = len(features)
    for f in features:
        data[("sqrt" + f)] = make_sqrt(data[f])
        data[("square" + f)] = make_square(data[f])
        data[("log" + f)] = make_log(data[f])

    for i in range(n):
        for j in range(i + 1, n):
            u = features[i]
            v = features[j]

            data[("geometric" + u + v)] = make_geometric(data[u], data[v])
            data[("harmonic" + u + v)] = make_harmonic(data[u], data[v])
            data[("quadratic" + u + v)] = make_quadratic(data[u], data[v])
            data[("sum" + u + v)] = make_sum(data[u], data[v])
            data[("diff" + u + v)] = make_diff(data[u], data[v])
            data[("recirocal" + u + v)] = make_recirocal(data[u], data[v])
            data[("recirocal" + v + u)] = make_recirocal(data[v], data[u])

    return data


def make_sqrt(x):
    return np.sqrt(x)


def make_square(x):
    return x ** 2


def make_log(x):
    return np.log1p(x)


def make_geometric(x, y):
    return np.sqrt(x * y)


def make_harmonic(x, y):
    return 2 / (1 / x + 1 / y)


def make_quadratic(x, y):
    return np.sqrt(x ** 2 + y ** 2) / 2


def make_sum(x, y):
    return x + y


def make_diff(x, y):
    return x - y


def make_recirocal(x, y):
    return x / y
