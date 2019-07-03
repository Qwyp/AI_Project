import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

n_neighbors = 15

#import some data to play with

iris = datasets.load_iris()

X = iris.data[:,:2]
y = iris.target

h = .02

cmap_light = ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000','#00FF00','#0000FF'])

for weights in ['uniform','distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X,y)

    x_min,x_max = X[:,0].min() -1, X[:,0].max()+1
    y_min,y_max = X[:,1].min() -1, X[:,1].max()+1

    xx,yy = np.meshgrid(np.arrange(x_min,x_max,h),np.arrange(y_min,y_max,h))
    Z = clf.predict(np.c[xx.ravel(),yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx,yy,Z,cmap=cmap_light)