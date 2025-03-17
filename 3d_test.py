# Import libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import math
 
 
# Create outer border
(xmin, xmax) = (0, 30)
(ymin, ymax) = (0, 20)

side_len = 1
# Coordinates of a square
x = lambda z: 5 + math.sin(z*2*math.pi)
y = lambda z: 3 + math.sin(z*2*math.pi)
square = lambda z: [[x(z),y(z), z], [x(z), y(z) + side_len, z], [x(z) + side_len, y(z) + side_len, z], [x(z) + side_len, y(z), z]]
zs = np.linspace(0, 10, 100)

# Coordinates of a second square
x2 = lambda z: 8 + math.sin(z*2*math.pi)
y2 = lambda z: 8 + math.sin(z*2*math.pi)
square2 = lambda z: [[x2(z),y2(z), z], [x2(z), y2(z) + side_len, z], [x2(z) + side_len, y2(z) + side_len, z], [x2(z) + side_len, y2(z), z]]

verts = []
for z in zs:
    verts.append(square(z))
    verts.append(square2(z))
#verts = np.array(verts)

ax = plt.figure().add_subplot(projection='3d')

poly = Poly3DCollection(verts, alpha=.7)
ax.add_collection3d(poly)
ax.set(xlim=(0, 10), ylim=(0, 12))

plt.show()
