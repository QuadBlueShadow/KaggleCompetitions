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
        """
        This is where we seperate the Passenger Ids into their two parts.\n
        The group they are in and the number of the person within that group.
        """

        # Get the column
        col = df["PassengerId"]

        # Make two new columns in the datafame and we will store the information here
        group = []
        group_num = []

        for i in range(len(col)):
            # We split the id and append to our lists
            info = col.iloc[i].split("_")
            group.append(int(info[0]))
            group_num.append(int(info[1]))

        # Create the new columns in the dataframe
        df["Group"] = group
        df["Group_PNum"] = group_num

    def convert_column_to_binary(self, df: pd.DataFrame, column: str):
        """
        This is used to turn true/false colums into 0/1 columns so that they can be interpereted by the models.
        """

        # Get the desired column
        col = df[column]

        # Determine if the new data needs to be 0 or 1
        adjusted = []
        for i in range(len(col)):
            adjusted.append(1 if col.iloc[i] == True else 0)

        # Replace the column with the new data
        df[column] = adjusted

    def get_mode(self, series: pd.Series):
        """
        This is used for groupby to find the mode of a certain column value in terms of another column.
        """

        # Drop nan values so that they are not considered in mode
        s = series.dropna()

        # If there is nothing but nans in the series just return nan
        if len(s) > 0:
            return s.mode()[0] # If there is more than one mode, just choose the first one
        return np.nan
    
    def get_mean(self, series: pd.Series):
        """
        This is used for groupby to find the mean of a certain column value in terms of another column.
        """

        # Drop nan values so that they are not considered in mean
        s = series.dropna()

        # If there is nothing but nans in the series just return nan
        if len(s) > 0:
            return round(s.mean(), 1) # Round so that it does not leave a ton of 0s after the decmial point
        return np.nan
    
    def get_average_spending(self, df: pd.DataFrame):
        """
        This is where the average spending of each passenger is calculated and added as a column to the dataframe.
        """

        total = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
        avg = total / 5
        df["AvgSpending"] = avg

    def fix_column_nans(self, df: pd.DataFrame, by = "Group", column = "HomePlanet", mean=False):
        """
        This function fixes nans by finding correlations in columns and filling in information that makes sense based on those correlations.
        """

        # Map values that may hvae nans in one column to the value the nan should be replaced with
        mapped = df.groupby(by)[column].agg(self.get_mode if not mean else self.get_mean)
        # Fix nans
        df[column] = df[column].fillna(df[by].map(mapped))

    def fill_na_col(self, df:pd.DataFrame, column: str, num=False):
        """
        This function just fills in nans with a default value.
        """
        df[column] = df[column].fillna("N/A" if not num else 0)

    def get_cabin_info(self, df: pd.DataFrame):
        """
        This function splits up the cabin colum into its three parts deck/num/side.
        """
        column = df["Cabin"]

        # New column info
        deck = []
        num = []
        side = []

        for i in range(len(column)):
            row = column.iloc[i]

            deck.append(ord(row[0]))
            num.append(int(row[2]))
            side.append(1 if row[4] == "P" else 0)

        # Add columns
        df["Deck"] = deck
        df["Num"] = num
        df["Side"] = side

    def get_proccessed_train_data(self, bins=5, test_bin=5):
        test_bin -= 1

        self.create_group_columns(self.train)

        self.fix_column_nans(self.train, by="Group", column="HomePlanet")
        self.fix_column_nans(self.train, by="Group", column="Cabin")
        self.fix_column_nans(self.train, by="Group", column="Destination")

        #self.fill_na_col(self.train, "CryoSleep", False)

        self.fix_column_nans(self.train, by="CryoSleep", column="RoomService", mean=True)
        self.fix_column_nans(self.train, by="CryoSleep", column="FoodCourt", mean=True)
        self.fix_column_nans(self.train, by="CryoSleep", column="ShoppingMall", mean=True)
        self.fix_column_nans(self.train, by="CryoSleep", column="Spa", mean=True)
        self.fix_column_nans(self.train, by="CryoSleep", column="VRDeck", mean=True)

        self.fix_column_nans(self.train, by="Age", column="RoomService", mean=False)
        self.fix_column_nans(self.train, by="Age", column="FoodCourt", mean=False)
        self.fix_column_nans(self.train, by="Age", column="ShoppingMall", mean=False)
        self.fix_column_nans(self.train, by="Age", column="Spa", mean=False)
        self.fix_column_nans(self.train, by="Age", column="VRDeck", mean=False)

        self.get_average_spending(self.train)

        self.fix_column_nans(self.train, by="AvgSpending", column="CryoSleep", mean=False)
        self.fix_column_nans(self.train, by="AvgSpending", column="Age", mean=False)

        self.train.dropna(inplace=True)
        #self.train["CryoSleep"].replace("N/A", np.nan)

        self.convert_column_to_binary(self.train, "CryoSleep")
        self.convert_column_to_binary(self.train, "VIP")
        self.convert_column_to_binary(self.train, "Transported")

        self.get_cabin_info(self.train)

        outgoing_data = self.train.loc[:, ("HomePlanet", "CryoSleep", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Group", "Group_PNum", "AvgSpending", "Deck", "Num", "Side")]
        partitioned_data = np.array_split(outgoing_data, bins)

        test_data = partitioned_data[test_bin]
        partitioned_data.pop(test_bin)

        train_data = np.concat(partitioned_data)

        return train_data, test_data
    
    def get_data(self, bins=5, test_bin=5):
        outgoing_data = self.train.loc[:, ("HomePlanet", "CryoSleep", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Group", "Group_PNum", "AvgSpending", "Deck", "Num", "Side")]
        partitioned_data = np.array_split(outgoing_data, bins)

        test_data = partitioned_data[test_bin]
        partitioned_data.pop(test_bin)

        train_data = np.concat(partitioned_data)

        return train_data, test_data