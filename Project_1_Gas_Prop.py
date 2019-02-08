# improt dependencies
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline


# read data
df_main = pd.read_csv('dataset.csv')
# let's peek into the dataset
df_main.head()
# describe the dataset
df_main.describe()
df_main.info()
# calculate the correlation matrix
corr = df_main.corr()

# plot the correlation heatmap
sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
plt.scatter(df_main['Idx'], df_main['Sv'], alpha=0.5)
plt.show(df_main['Idx'], df_main['Sv'], alpha=0.5)
plt.scatter(df_main['Idx'], df_main['Th'], alpha=0.5)
plt.show(df_main['Idx'], df_main['Th'], alpha=0.5)
columns_x = ['Th', 'Sv', 'Tm', ' Pr']
column_label = ['Idx']
X = df_main[columns_x]
y = df_main[column_label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)
reg = LinearRegression()
reg = reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
r2_score(y_test, y_pred)
# other way of calculating the R2 score
reg.score(X_test, y_test)
print("weights: ",reg.coef_)
print("bias: ",reg.intercept_)
# split array in k(number of folds) sub arrays
X_folds = np.array_split(X_train, 3)
y_folds = np.array_split(y_train, 3)
scores = list()
models = list()
for k in range(3):
    reg = LinearRegression()

    # We use 'list' to copy, in order to 'pop' later on
    X_train_fold = list(X_folds)
    # pop out kth sub array for testing
    X_test_fold  = X_train_fold.pop(k)
    # concatenate remaining sub arrays for training
    X_train_fold = np.concatenate(X_train_fold)

    # same process for y
    y_train_fold = list(y_folds)
    y_test_fold  = y_train_fold.pop(k)
    y_train_fold = np.concatenate(y_train_fold)

    reg = reg.fit(X_train_fold, y_train_fold)
    scores.append(reg.score(X_test_fold, y_test_fold))
    models.append(reg)

print(scores)
# polynomial model
for count, degree in enumerate([2, 3, 4, 5, 6]):
    print("Degree ",degree)
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("R2 score: ",r2_score(y_test, y_pred))
    print("coefficiets: ",model.steps[1][1].coef_)
    print("bias: ",model.steps[1][1].intercept_)
    print("---------------------------------")
    print()
