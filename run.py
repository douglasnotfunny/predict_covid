import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
from sklearn import svm
import numpy as np


df = pd.read_csv('data.csv')
all_columns = df.columns

index_start = 0
count = 1
index_start = 0

for d in df['location']:
    if d=="World" and index_start == 0:
        index_start = count + 1
    if index_start!=0 and d!= "World":
        index_end = count - 1
        break
    count+=1

print(index_start, index_end)

df = df.iloc[index_start:index_end]
print(df)

'''df_types = df.dtypes
aggregation_columns = {}
count = 0

for d in df_types:
    name_column = all_columns[count]
    if d == 'float64':
        aggregation_columns[name_column] = str(d)
    count+=1

df = df.groupby(by=["location"]).sum()

for f in df.columns: 
    if df[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(df[f].values)) 
        df[f] = lbl.transform(list(df[f].values))

print(df.head())

y = df["new_cases"].fillna(0)
X = df.drop(['new_cases'], axis=1).fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

xg_reg = svm.SVR()
xg_reg.fit(X_train, y_train)

score = xg_reg.score(X_test, y_test)
preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"RMSE: {rmse}")
print(f"Score: {score}")
print(f"Predição {preds}")'''