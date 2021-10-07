# importar bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb
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

y = df["new_cases"]
X = df.drop(['new_cases'], axis=1)

print(f"{X}\n\n\n")
print(f"{y}")

print(X.shape,y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

xg_reg = xgb.XGBRegressor(n_estimators = 1000, learning_rate=0.05)

