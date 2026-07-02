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

    def fix_planet_nans(self, df: pd.DataFrame):
        for i in range(len(df)):
            row = df.iloc[i]

            if type(row["HomePlanet"]) == float:
                group = row["Group"]

                others = df[(df["Group"] == group) & (~df["HomePlanet"].isna())]
                if len(others) > 0:
                    planet = others.iloc[0]["HomePlanet"]
                else:
                    planet = "__MISSING__"

                df.loc[i, "HomePlanet"] = planet

    def get_proccessed_train_data(self):
        self.create_group_columns(self.train)

        self.convert_column_to_binary(self.train, "CryoSleep")
        self.convert_column_to_binary(self.train,"VIP")
        self.convert_column_to_binary(self.train, "Transported")

        self.fix_planet_nans(self.train)

        return [], []