import pandas as pd
import numpy as np
import statistics
import time

class DataFeeder:
    def __init__(self, train_file_path = "", sub_file_path = ""):
        self.train = pd.read_csv(train_file_path)
        if sub_file_path != "":
            self.sumbission_data = pd.read_csv(sub_file_path)
        
        self.cols = self.train.columns

    def create_group_columns(self, df: pd.DataFrame):
        col = df["PassengerId"]

        group = []
        group_num = []
        for i in range(len(col)):
            info = col.iloc[i].split("_")
            group.append(int(info[0]))
            group_num.append(int(info[1]))

        df["Group"] = group
        df["Group_PNum"] = group_num

    def convert_column_to_binary(self, df: pd.DataFrame, column: str):
        col = df[column]

        adjusted = []
        for i in range(len(col)):
            adjusted.append(1 if col.iloc[i] == True else 0)

        df[column] = adjusted

    def get_mode(self, series: pd.Series):
        s = series.dropna()

        if len(s) > 0:
            return s.mode() 
        return np.nan
    
    def get_mean(self, series: pd.Series):
        s = series.dropna()

        if len(s) > 0:
            return s.mean() if s.mean() != 0.0 else 0.0
        return np.nan
    
    def get_average_spending(self, df: pd.DataFrame):
        total = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
        avg = total / 5
        df["AvgSpending"] = avg

    def fix_column_nans(self, df: pd.DataFrame, by = "Group", column = "HomePlanet", mean=False):
        mapped = df.groupby(by)[column].agg(self.get_mode if not mean else self.get_mean)
        if mean:
            print(mapped)
        df[column] = df[column].fillna(df[by].map(mapped))

    def fill_na_col(self, df:pd.DataFrame, column: str, num=False):
        df[column] = df[column].fillna("N/A" if not num else 0)

    def get_proccessed_train_data(self, bins=5):
        self.create_group_columns(self.train)

        self.fix_column_nans(self.train, by="Group", column="HomePlanet")
        self.fix_column_nans(self.train, by="Group", column="Cabin")
        self.fix_column_nans(self.train, by="Group", column="Destination")
        self.fix_column_nans(self.train, by="CryoSleep", column="RoomService", mean=True)
        # self.fix_column_nans(self.train, by="CryoSleep", column="FoodCourt", mean=True)
        # self.fix_column_nans(self.train, by="CryoSleep", column="ShoppingMall", mean=True)
        # self.fix_column_nans(self.train, by="CryoSleep", column="Spa", mean=True)
        # self.fix_column_nans(self.train, by="CryoSleep", column="VRDeck", mean=True)

        self.get_average_spending(self.train)

        # self.convert_column_to_binary(self.train, "CryoSleep")
        # self.convert_column_to_binary(self.train, "VIP")
        # self.convert_column_to_binary(self.train, "Transported")

        return [], []