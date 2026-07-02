import pandas as pd
import numpy as np
import statistics

from sklearn.linear_model import LogisticRegression
from sklearn import svm

from Helpers.data_managment import DataFeeder

import time

df = DataFeeder("SpaceshipTitanic/Data/train.csv")
df.get_proccessed_train_data()

print(df.train.head())