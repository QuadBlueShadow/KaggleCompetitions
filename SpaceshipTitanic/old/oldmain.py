import pandas as pd
import numpy as np
import statistics

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from SpaceshipTitanic.old.util import LogReg

import time

alphabet_values = {
    "A": 0.1,
    "B": 0.2,
    "C": 0.3,
    "D": 0.4,
    "E": 0.5,
    "F": 0.6,
    "G": 0.7,
    "H": 0.8,
    "I": 0.9,
    "J": 1.0,
    "K": 1.1,
    "L": 1.2,
    "M": 1.3,
    "N": 1.4,
    "O": 1.5,
    "P": 1.6,
    "Q": 1.7,
    "R": 1.8,
    "S": 1.9,
    "T": 2.0,
    "U": 2.1,
    "V": 2.2,
    "W": 2.3,
    "X": 2.4,
    "Y": 2.5,
    "Z": 2.6
}

def convert_bool_to_num(df, column):
    tmp = [0 if val == False else 1 for val in df[column]]

    df[column] = tmp

def normalize_column(df, column, na_fill=-1):
    df[column] = df[column].fillna(na_fill)
    data = df[column]

    mean = statistics.mean(data)
    std = statistics.stdev(data)

    data = (data - mean) / std

    df[column] = data

def fill_nan(df, column, val):
    df[column] = df[column].fillna(val)

def transform_cabin(df:pd.DataFrame, nan_fill):
    df["Cabin"] = df["Cabin"].fillna(nan_fill)

    part1 = []
    part2 = []
    part3 = []

    for i in range(len(df["Cabin"])):
        if df["Cabin"].iloc[i] == "None":
            part1.append(-0.1)
            part2.append(-0.1)
            part3.append(-0.1)
        else:
            c = df["Cabin"].iloc[i].split("/")
            part1.append(alphabet_values[c[0]])
            part2.append(int(c[1])/5)
            part3.append(alphabet_values[c[2]])

    df["Part1"] = part1
    df["Part2"] = part2
    df["Part3"] = part3

full_data = pd.read_csv("SpaceshipTitanic/Data/train.csv")
challenge_data = pd.read_csv("SpaceshipTitanic/Data/test.csv")

convert_bool_to_num(full_data, "Transported")
convert_bool_to_num(full_data, "VIP")
convert_bool_to_num(full_data, "CryoSleep")

normalize_column(full_data, "RoomService")
normalize_column(full_data, "FoodCourt")
normalize_column(full_data, "ShoppingMall")
normalize_column(full_data, "Spa")
normalize_column(full_data, "VRDeck")
normalize_column(full_data, "Age")

fill_nan(full_data, "HomePlanet", "None")
fill_nan(full_data, "Cabin", "None")
fill_nan(full_data, "Destination", "None")

transform_cabin(full_data, "None")

unique_planets = full_data["HomePlanet"].unique()
print(unique_planets)

planet_dict = {}

for i in range(len(unique_planets)):
    planet_dict[unique_planets[i]] = i

print(planet_dict)


unique_destinations = full_data["Destination"].unique()
print(unique_destinations)

destination_dict = {}

for i in range(len(unique_destinations)):
    destination_dict[unique_destinations[i]] = i

print(destination_dict)


#time.sleep(10000)

print(len(full_data))
training_data = full_data.iloc[0:5500]
testing_data = full_data.iloc[5500:]

print(training_data.head())


x = []

for i in range(len(training_data)):
    row = training_data.iloc[i]
    x.append([planet_dict[row["HomePlanet"]], destination_dict[row["Destination"]], row["Age"], row["VIP"], row["RoomService"], row["FoodCourt"], row["ShoppingMall"], row["Spa"], row["VRDeck"], row["Part1"], row["Part3"]])

x_test = []

for i in range(len(testing_data)):
    row = testing_data.iloc[i]
    x_test.append([planet_dict[row["HomePlanet"]], destination_dict[row["Destination"]], row["Age"], row["VIP"], row["RoomService"], row["FoodCourt"], row["ShoppingMall"], row["Spa"], row["VRDeck"], row["Part1"], row["Part3"]])

model = LogisticRegression(max_iter=300).fit(np.array(x), np.array(training_data["Transported"]))
# model = LogReg(7)
# model.train(np.array(x), np.array(training_data["Transported"]))

clf = svm.SVC()
clf.fit(np.array(x), np.array(training_data["Transported"]))



prediction = model.predict(np.array(x_test))
preds = clf.predict(np.array(x_test))

#print(model.weights, model.b)

score = 0
score2 = 0
for i in range(len(prediction)):
    row = testing_data.iloc[i]

    if prediction[i] == row["Transported"]:
        score += 1

    if preds[i] == row["Transported"]:
        score2 += 1

print("Accuracy:", score/len(prediction))
print("Accuracy 2:", score2/len(prediction))


# rows_with_nans = challenge_data[challenge_data.isna().any(axis=1)]
# print(rows_with_nans)


convert_bool_to_num(challenge_data, "VIP")
convert_bool_to_num(challenge_data, "CryoSleep")

normalize_column(challenge_data, "RoomService")
normalize_column(challenge_data, "FoodCourt")
normalize_column(challenge_data, "ShoppingMall")
normalize_column(challenge_data, "Spa")
normalize_column(challenge_data, "VRDeck")
normalize_column(challenge_data, "Age")

fill_nan(challenge_data, "HomePlanet", "None")
fill_nan(challenge_data, "Cabin", "None")
fill_nan(challenge_data, "Destination", "None")

transform_cabin(challenge_data, "None")

print(challenge_data.head())



x = []
indicies = []

for i in range(len(challenge_data)):
    row = challenge_data.iloc[i]
    indicies.append(row["PassengerId"])
    x.append([planet_dict[row["HomePlanet"]], destination_dict[row["Destination"]], row["Age"], row["VIP"], row["RoomService"], row["FoodCourt"], row["ShoppingMall"], row["Spa"], row["VRDeck"], row["Part1"], row["Part3"]])

preds = clf.predict(np.array(x))

data = []

for i in range(len(preds)):
    data.append([indicies[i], True if preds[i] == 1 else False])

df = pd.DataFrame(data=data, columns=["PassengerId", "Transported"])
print(df.columns)
df.set_index(["PassengerId"], inplace=True)

print(df.head())

df.to_csv("Results.csv")