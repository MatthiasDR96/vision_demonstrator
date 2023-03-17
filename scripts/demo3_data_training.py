# Imports
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from sklearn import svm

# Load data
df = pd.read_csv("data/color_data.csv")

# Encode categorical labels
labelencoder= LabelEncoder() 
df['Class'] = labelencoder.fit_transform(df['Class'])

df.fillna(0, inplace=True)

# Train
clf = svm.SVC()
clf.fit(df[['H', 'S', 'V']].values, df['Class'].values)

# Save model
filename = 'data/model.sav'
pickle.dump(clf, open(filename, 'wb'))