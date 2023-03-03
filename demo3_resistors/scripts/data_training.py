# Imports
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from sklearn import svm

# Load data
df = pd.read_csv("demo3_resistors/color_data")

# Encode categorical labels
labelencoder= LabelEncoder() 
df['Class'] = labelencoder.fit_transform(df['Class'])

# Train
clf = svm.SVC()
clf.fit(df[['R', 'G', 'B']].values, df['Class'].values)

# Save model
filename = 'demo3_resistors/model.sav'
pickle.dump(clf, open(filename, 'wb'))