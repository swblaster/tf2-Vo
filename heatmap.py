import numpy as np
import matplotlib
import matplotlib.pyplot as plt

y = [0.005, 0.01, 0.05, 0.1, 0.5]
x = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

mydata = [[98.0, 96.8, 98.0, 96.4, 93.0],
		  [98.2, 98.6, 98.0, 98.0, 97.4],
		  [98.6, 98.2, 98.2, 98.0, 98.2],
		  [98.4, 98.4, 98.6, 98.0, 98.2],
		  [98.4, 98.6, 98.4, 98.2, 98.0],
		  [98.4, 98.2, 98.2, 98.2, 97.6],
          [98.2, 98.2, 98.2, 98.0, 96.0],
		  [98.4, 98.0, 98.2, 98.0, 94.3],
		  [97.8, 97.6, 97.8, 98.0, 91.0],
		  [94.3, 95.0, 95.9, 97.8, 90.3]]
data = np.zeros((5, 10))
for i in range (10):
    for j in range(5):
        data[j][i] = mydata[i][j]
print (data)

fig, ax = plt.subplots()
im = ax.imshow(data, cmap=plt.cm.coolwarm)
ax.set_xticks(np.arange(len(x)), minor=False)
ax.set_yticks(np.arange(len(y)), minor=False)
ax.set_xticklabels(x)
ax.set_yticklabels(y)

for i in range(len(x)):
    for j in range(len(y)):
        text = ax.text(i, j, data[j, i],
                       ha="center", va="center", color="w")
fig.tight_layout()
plt.colorbar(im, shrink=0.5, aspect=20*0.7)
plt.show()
