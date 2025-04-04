# Import Data manipulation Libraries

import pandas as pd
import numpy as np

# Import Data Visualization Libraries

import seaborn as sns
import matplotlib.pyplot as plt

# import filter warnings library

import warnings
warnings.filterwarnings('ignore')

# import logging library

import logging
logging.basicConfig(filename = "model.log",
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load the dataset

D1 = ('https://raw.githubusercontent.com/adnansayyedd/Energy_Consumption_Prediction_MLModel/refs/heads/main/train_energy_data.csv')
D2 = ('https://raw.githubusercontent.com/adnansayyedd/Energy_Consumption_Prediction_MLModel/refs/heads/main/test_energy_data.csv')

df1 = pd.read_csv(D1)
df2 = pd.read_csv(D2)

# Combining both datasets

df = pd.concat([df1, df2], axis = 0)

# Label Encoding categorical variables
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

df['Building Type'] = LE.fit_transform(df['Building Type'])

df['Day of Week'] = LE.fit_transform(df['Day of Week'])

# Splitting the dataset into X and y

from sklearn.model_selection import train_test_split

X = df.drop('Energy Consumption', axis = 1)
y = df['Energy Consumption']

# Splitting the dataset into training set and test set

from  sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# scaling the features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# import the random forest regressor model

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

RF = RandomForestRegressor()

RF.fit(X_train, y_train)

y_pred_RF = RF.predict(X_test)

r2_score_RF = r2_score(y_test, y_pred_RF)

r2_score_RF

print(f'Random Forest Regressor R2 Score: {r2_score_RF}')