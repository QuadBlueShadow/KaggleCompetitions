import pandas as pd
import numpy as np
import statistics

from sklearn.linear_model import LogisticRegression
from sklearn import svm

from Helpers.data_managment import DataFeeder

df = DataFeeder("SpaceshipTitanic/Data/train.csv")
print(df.get_proccessed_train_data(bins=5, test_bin=5)[1].shape)

