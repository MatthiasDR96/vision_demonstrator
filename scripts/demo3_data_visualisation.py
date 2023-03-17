import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
df = pd.read_csv("data/color_data.csv")

c=df['Class'].map({'x':'gray','r':'r','z':'brown','k':'k','b':'b','v':'m','g':'g'})

# Set figure
fig = plt.figure(figsize=(12, 9))
ax = Axes3D(fig)

# Plot data
y = df['H']
x = df['S']
z = df['V']
ax.scatter(x,y,z, c=c)

# Set axis
ax.set_ylabel('H')
ax.set_xlabel('S')
ax.set_zlabel('V')

# Show
plt.show()