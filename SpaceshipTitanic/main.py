import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from xgboost import XGBClassifier

from Helpers.data_managment import DataFeeder
from Helpers.model_testing import cross_validate

from catboost import CatBoostClassifier

df = DataFeeder("SpaceshipTitanic/Data/train.csv", "SpaceshipTitanic/Data/test.csv")
df.set_proccessed_train_data()
submission_data = df.get_proccessed_submission_data()

LogisticReg = LogisticRegression(max_iter=2000)
params = LogisticReg.get_params()
print("Logistic Regression")
cross_validate(LogisticRegression, df, True, 5, **params)
print()

SupportVecMachine = svm.SVC(max_iter=7000)
params = SupportVecMachine.get_params()
print("SVM")
cross_validate(svm.SVC, df, True, 5, **params)
print()

xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=8,
    random_state=42
)
params = xgb.get_params()
print("XGBoost")
cross_validate(XGBClassifier, df, False, 5, **params)
print()

cbc = CatBoostClassifier(
    iterations=900,
    learning_rate=0.01,
    depth=10,
    random_seed=42,
    verbose=0          # Suppresses training output
)
params = cbc.get_params()
print("CatBoost")
best = cross_validate(CatBoostClassifier, df, False, 5, **params)
print()

preds = best.predict(submission_data.to_numpy())

preds = [True if p else False for p in preds]

output = pd.DataFrame()
output["PassengerId"] = df.sumbission_data["PassengerId"]
output["Transported"] = preds

output.set_index("PassengerId", inplace=True)

print(output.head(5))

output.to_csv("Results.csv")