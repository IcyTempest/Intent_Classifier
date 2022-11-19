import pandas as pd
import numpy as np

from domain.domain import *

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import metrics

# Load Data
df = pd.read_csv("all_segment_dataset.csv")

# Vectorize/Embed the Data using FastText
# Encode Intent using LabelEncoder
utility: Utility = Utility(df["Question"], df["Intent"])

print(type(utility.embeddedX))
print(type(utility.encodedY))
dff = pd.DataFrame({
    "Question": utility.embeddedX, "Intent": utility.encodedY.tolist(),
})
print(type(dff["Question"].to_numpy()))

utility.embeddedX = dff["Question"].to_numpy()
# utility.encodedY = dff["Intent"]

print(utility.embeddedX[0])


myShuffle = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
myShuffle.get_n_splits(dff["Question"], dff["Intent"])
X_train, X_test, Y_train, Y_test = None, None, None, None

for train_index, test_index in myShuffle.split(utility.embeddedX, utility.encodedY):
    print("train_index: ",train_index)
    X_train, X_test = utility.embeddedX[train_index], utility.embeddedX[test_index]
    Y_train, Y_test = utility.encodedY[train_index], utility.encodedY[test_index]

print("x_train: ", len(X_train))
print("x_test: ", len(X_test))
print("y_train: ", len(Y_train))
print("y_test: ", len(Y_test))


# Split data into 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(utility.embeddedX, utility.encodedY, test_size=0.2, random_state=42)


# ----- Logistic Regression ------------------------------ #
# --- Multinomial and Newton-CG Solver and l2 penality --- #
def MyLog():
    myLog = LogisticRegression(multi_class="multinomial", solver="newton-cg", max_iter=300, penalty="l2")
    myLog.fit(X_train, Y_train)

    score = myLog.score(X_test, Y_test)
    y_pred = myLog.predict(X_test)

    print("Test Score: ", score)
    utility.getScore(Y_pred=y_pred, Y_test=Y_test, average="weighted")
    utility.getScore(Y_pred=y_pred, Y_test=Y_test, average="binary")

# -------------------------------------------------- #

# MyLog()
