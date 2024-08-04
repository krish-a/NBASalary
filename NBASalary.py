import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def stats(): 
  stats = pd.read_csv("stats.csv")
  stats = stats[["Player", "Age", "G", "FG%", "3P%", "2P%", "eFG%", "FT%", "TRB", "AST", "STL", "BLK", "PTS"]]
  agg_funcs = {'Age': 'max','G': 'sum','FG%': 'mean','3P%': 'mean','2P%': 'mean','eFG%': 'mean','FT%': 'mean',
      'TRB': 'mean','AST': 'mean','STL': 'mean','BLK': 'mean','PTS': 'mean'}
  stats = stats.groupby('Player').agg(agg_funcs).reset_index()
  stats.iloc[:, 1:] = stats.iloc[:, 1:].round(2)
  stats = stats.dropna()
  salary = pd.read_csv("salary.csv")
  salary = salary[["Player", "2024-25"]]
  salary = salary.rename(columns={'2024-25': 'Salary'})
  salary = salary.dropna()
  df = stats.merge(salary, on='Player', how='inner')
  df.dropna(subset=['Salary'])
  df['Salary'] = df['Salary'].replace('[\$,]', '', regex=True).astype(int)

  data = df.drop(columns=['Player'])
  X = data.drop(columns=['Salary'])
  y = data['Salary']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  #print(f'Mean Squared Error: {mse}')
  #print(f'R^2 Score: {r2}')


  
