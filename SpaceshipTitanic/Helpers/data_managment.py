import pandas as pd
import numpy as np
import statistics

class DataFeeder:
    def __init__(self, file_path=""):
        self.data = pd.read_csv(file_path)
        
        self.cols = self.data.columns

    def create_group_column(self):
        col = self.data["PassengerId"]

        group = []
        group_num = []
        for i in range(len(col)):
            info = col.iloc[i].split("_")
            group.append(int(info[0]))
            group_num.append(int(info[1]))

        self.data["Group"] = group
        self.data["Group"]

    def get_proccessed_data(self):

        return [], []