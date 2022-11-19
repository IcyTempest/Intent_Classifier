import pandas as pd
import numpy as np

from domain.domain import *

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import metrics

from domain.logistic_regression import MyLog

# Load Data
df = pd.read_csv("all_segment_dataset.csv")

# Vectorize/Embed the Data using FastText
# Encode Intent using LabelEncoder
utility: Utility = Utility(df["Question"], df["Intent"])

# Split data into 20% testing
X_train, X_test, Y_train, Y_test = Utility.train_test_split_with_StratifiedShuffleSplit(X=utility.embeddedX,
                                                                                        Y=utility.encodedY,
                                                                                        test_size=0.2,
                                                                                        n_splits=5)

# # Split data into 20% testing using classic train_test_split, doesn't work
# X_train, X_test, Y_train, Y_test = train_test_split(utility.embeddedX, utility.encodedY, test_size=0.2, random_state=42)


# ----- Logistic Regression ------------------------------ #
myLog: MyLog = MyLog(X_train, X_test, Y_train, Y_test, utility)
myLog.run()
# -------------------------------------------------- #
