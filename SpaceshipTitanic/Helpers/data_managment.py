import pandas as pd
import numpy as np
import statistics

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

    def fix_planet_nans(self, df: pd.DataFrame):
        nans = df[df["HomePlanet"].isna()]

        for i in range(len(nans)):
            id = nans.iloc[i]["PassengerId"]
            group = df[df["PassengerId"] == id]["Group"]

            print(group)

            planet = df[(df["Group"] == group) & (~df["HomePlanet"].isna())].iloc[i]["HomePlanet"]
            df[df["PassengerId"] == id]["HomePlanet"] = planet
        return nans

    def get_proccessed_train_data(self):
        self.create_group_columns(self.train)

        self.convert_column_to_binary(self.train, "CryoSleep")
        self.convert_column_to_binary(self.train,"VIP")
        self.convert_column_to_binary(self.train, "Transported")

        nan_rows = self.fix_planet_nans(self.train)
        print(nan_rows)

        return [], []