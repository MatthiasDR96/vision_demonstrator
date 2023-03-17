# Imports 
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vision_demonstrator.preprocessing import *

# Load data
df = pd.read_csv("data/color_data.csv")
c=df['Class'].map({'x':'gray','r':'red','z':'brown','k':'black','b':'blue','v':'magenta','g':'green'})

# Plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.H, df.S, df.V, c=c, alpha=.6, edgecolor='k', lw=0.3)
ax.set_xlabel('H', fontsize=14)
ax.set_ylabel('S', fontsize=14)
ax.set_zlabel('V', fontsize=14)
plt.show()