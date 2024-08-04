import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Function to predict salary for a new player
# Example usage:
# new_player = {
#     'Age': 25, 'G': 82, 'FG%': 0.480, '3P%': 0.380, '2P%': 0.520,
#     'eFG%': 0.550, 'FT%': 0.850, 'TRB': 6.0, 'AST': 5.0,
#     'STL': 1.5, 'BLK': 0.5, 'PTS': 20.0
# }
# predicted_salary = predict_salary(new_player)
# print(f"Predicted Salary: ${predicted_salary:.2f}")
def predict_salary(new_player_stats):
    new_player_df = pd.DataFrame([new_player_stats])
    new_player_scaled = scaler.transform(new_player_df)
    predicted_salary = model.predict(new_player_scaled)[0]
    return predicted_salary

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

  '''data = df.drop(columns=['Player'])
  X = data.drop(columns=['Salary'])
  y = data['Salary']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)'''
  #print(f'Mean Squared Error: {mse}')
  #print(f'R^2 Score: {r2}')
    
  X = data.drop(['Salary'], axis=1)
  y = data['Salary']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Scale the features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # Create and train the model
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train_scaled, y_train)

  # Make predictions on test set
  y_pred = model.predict(X_test_scaled)

  # Evaluate model
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_test, y_pred)

  # Feature importance
  feature_importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
  feature_importance = feature_importance.sort_values('importance', ascending=False)


  
