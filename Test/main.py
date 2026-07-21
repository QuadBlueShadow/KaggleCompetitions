import matplotlib.pyplot as plt
import numpy as np

# Create the figure and add a 3D subplot
fig = plt.figure()
ax = plt.axes(projection='3d')

# Generate sample data
z = np.linspace(0, 15, 100)
x = np.sin(z)
y = np.cos(z)

# Plot a 3D line
ax.plot3D(x, y, z, 'gray')
plt.show()