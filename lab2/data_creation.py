from sklearn.model_selection import train_test_split
import os
import pandas as pd


def creation_data():
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
                     delimiter=';')
    df_train, df_classes = df.drop(columns=['quality']), df['quality']
    X_train, X_test, y_train, y_test = train_test_split(df_train, df_classes, test_size=0.2, random_state=42)
    if os.path.isdir('train') is False:
        os.mkdir("train")
    if os.path.isdir('test') is False:
        os.mkdir("test")
    X_train.to_csv('train/X_train.csv')
    X_test.to_csv('test/X_test.csv')
    y_train.to_csv('train/y_train.csv')
    y_test.to_csv('test/y_test.csv')
