import os
import pandas as pd
from preprocessing import get_X_y
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.metrics import f1_score , precision_score , recall_score

csv_path = os.path.join(
    os.path.dirname(__file__) ,
    ".." ,
    "dataset" ,
    "covtype.csv"
)

df = pd.read_csv(csv_path)

X , y = get_X_y(df)

X_train , X_test , y_train , y_test = train_test_split(
    X ,
    y ,
    test_size = 0.2 ,
    random_state = 42 ,
    stratify = y
)

weights = compute_sample_weight(
    class_weight = "balanced" ,
    y = y_train
)

model = XGBClassifier(
    subsample = 0.8 ,
    colsample_bytree = 0.8 ,
    random_state = 42 ,
    n_jobs = -1
)

model.fit(
    X_train ,
    y_train ,
    sample_weight = weights
)

y_pred = model.predict(X_test)


print("F1:", f1_score(y_test, y_pred, average="macro"))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall:", recall_score(y_test, y_pred, average="macro"))