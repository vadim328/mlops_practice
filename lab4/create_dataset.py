from catboost.datasets import titanic
import pandas

titanic_train = titanic()[0]
titanic_train.to_csv(path_or_buf = "datasets/titanic.csv")
