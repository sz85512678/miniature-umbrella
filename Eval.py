import Features
import Models
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import numpy as np
from sklearn.model_selection import KFold


def evaluate(df, cv=KFold(),
             features=Features.make_features_naive,
             model1=Models.MyDummyClassifier(),
             model2=Models.MyDummyClassifier(),
             metric=balanced_accuracy_score,
             ):
    err1s = []
    err2s = []
    for i, (train_index, test_index) in enumerate(cv.split(df)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")

        df_train = df.iloc[train_index]
        df_test = df.iloc[test_index]
        X_train, y1_train, y2_train = features(df_train)
        X_test, y1_test, y2_test = features(df_test)

        model1.fit(X_train, y1_train)
        y1_train_pred = model1.predict(X_train)
        y1_train_err = metric(y1_train, y1_train_pred)
        print(f"  Train err1: {y1_train_err}")

        y1_pred = model1.predict(X_test)
        err1 = metric(y1_test, y1_pred)
        err1s.append(err1)
        print(f"  Test err1: {err1}")

        model2.fit(X_train, y2_train)
        y2_train_pred = model2.predict(X_train)
        y2_train_err = metric(y2_train, y2_train_pred)
        print(f"  Train err2: {y2_train_err}")

        y2_pred = model2.predict(X_test)
        err2 = metric(y2_test, y2_pred)
        err2s.append(err2)
        print(f"  Test err2: {err2}")

    print(f"Mean err1: {np.mean(err1s)}")
    print(f"Mean err2: {np.mean(err2s)}")

    return err1s, err2s
