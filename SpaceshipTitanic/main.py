import pandas as pd
import numpy as np
import statistics

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from SpaceshipTitanic.old.util import LogReg

from Helpers.data_managment import DataFeeder

import time

df = DataFeeder("SpaceshipTitanic/Data/train.csv")