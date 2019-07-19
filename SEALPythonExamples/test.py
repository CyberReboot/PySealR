import time
import random
import pickle
import threading
import numpy as np
import pandas as pd
import math
import random
import seal

#importing the dataset
dataset = pd.read_csv ('Social_Network_Ads.csv')
X = dataset.iloc [:, [2, 3]].values
Y = dataset.iloc [:, 4].values
print (dataset)

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.2, random_state = 0)

print (X)
