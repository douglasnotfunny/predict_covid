# importar bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing


# importar train.csv em DataFrame
df = pd.read_csv('data.csv')

# visualizar as 5 primeiras entradas
df.head()

all_collumns = df.columns

df = pd.DataFrame(df)

for f in df.columns: 
    if df[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(df[f].values)) 
        df[f] = lbl.transform(list(df[f].values))


# separar entre as variáveis X e y
X = all_collumns.drop("new_cases")
y = df["new_cases"]
