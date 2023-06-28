import pandas as pd


titanic = pd.read_csv("datasets/titanic.csv")
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean().round(0))
titanic.to_csv("datasets/titanic.csv")
