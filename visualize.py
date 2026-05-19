from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2026/01/08 15:45:02

@author: Javiera Jilberto Vallejos 
'''

v1 = [-0.367563,   -0.92452162, -0.10078301]
v2 = [0.57849547, -0.80836495, -0.10903716]
v3 = [ 0.10283977, -0.98930911, -0.10339951]

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

origin = [0, 0, 0]
vectors = [v1, v2, v3]
colors = ['r', 'g', 'b']
labels = ['v1', 'v2', 'v3']

for vec, color, label in zip(vectors, colors, labels):
    ax.quiver(*origin, *vec, color=color, label=label, arrow_length_ratio=0.1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('3D Vector Visualization')
plt.show()