# NBA Player Salary Predictor
## Overview
This repository contains Python code to predict salaries for NBA players based on their stats using machine learning. It's geared towards role players rather than star players who negotiate higher salaries based on their reputation.

## How It Works
- Scrapes player stats and salary data from websites using requests and BeautifulSoup libraries.
- Parses and cleans the scraped data, then loads it into pandas dataframes.
- Aggregates stats per player and merges them with salary info.
- Splits data into training and testing sets.
- Standardizes features and trains models like RandomForest and GradientBoosting using optimal settings found via grid search.
- Uses the best model to predict player salaries.
- Evaluates predictions and displays key stats like RMSE and R-squared.

## Notes
- This model predicts salaries based on player stats, mainly suitable for NBA role players.
- Star players with established market value might not fit predictions accurately.

