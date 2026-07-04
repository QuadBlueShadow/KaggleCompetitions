import pandas as pd
import numpy as np
import statistics

from sklearn.linear_model import LogisticRegression
from sklearn import svm

from Helpers.data_managment import DataFeeder

import time

df = DataFeeder("SpaceshipTitanic/Data/train.csv")
df.get_proccessed_train_data(bins=5)

# print(df.train[df.train.isna().any(axis=1)])
# print(df.train[df.train["CryoSleep"].isna()])
# print(df.train[df.train["VIP"].isna()])

print(df.train.head(50))
# nans = df.train[df.train["HomePlanet"].isna()]
# print(nans)