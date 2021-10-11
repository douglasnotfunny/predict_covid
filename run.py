import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
all_columns = df.columns

index_start = 0
count = 1
index_end = 0

for d in df['location']:
    if d=="World" and index_start == 0:
        index_start = count + 1
    if index_start!=0 and d!= "World":
        index_end = count - 1
        break
    count+=1

print(index_start, index_end)

df = df.iloc[index_end-30:index_end]
df.to_csv('out.csv')

for f in df.columns: 
    if df[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(df[f].values)) 
        df[f] = lbl.transform(list(df[f].values))

y = df["new_cases"].fillna(0)
X = df.drop(['new_cases','iso_code','continent','location','date','total_cases','total_deaths','total_cases_per_million','total_deaths_per_million'], axis=1).fillna(0)

next_value = 0
count = 0

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)

    xg_reg = xgb.XGBRegressor(n_estimators = 300, learning_rate=0.01)
    xg_reg.fit(X_train, y_train)

    score = xg_reg.score(X_test, y_test)
    preds = xg_reg.predict(X_test)

    r2 = r2_score(y_test, preds)
    if score > 0.90:
        count+=1
        print(f"Score: {score}")
        print("R2: %f" % (r2))
        next_value+=preds.mean()

next_value = next_value/count
print(f"Predição Média: {next_value}")