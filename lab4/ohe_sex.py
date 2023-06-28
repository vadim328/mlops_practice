import pandas as pd
from sklearn.preprocessing import OneHotEncoder


titanic = pd.read_csv("datasets/titanic.csv")

onehotencoder = OneHotEncoder(sparse_output = False)
encoded_df = pd.DataFrame(onehotencoder.fit_transform(titanic[["Sex"]]))
titanic = titanic.join(encoded_df)

titanic.to_csv("datasets/titanic.csv")
