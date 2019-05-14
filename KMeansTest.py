# -*- coding: utf-8 -*-
"""

@author: Ramon
"""

from skimage import io
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt
import time

import KMeans as km


plt.close("all")
if __name__ == "__main__":
    import os
    print(os.path.dirname(os.path.realpath(__file__)))
    im = io.imread('Images/0053.jpg')
    # plt.figure(1)
    # plt.imshow(im)
    # plt.axis('off')
    # plt.show()
    options = {'verbose': False, 'km_init': 'first'}
    k_m = km.KMeans(im, 3, options)
    k_m.run()
    print(k_m.centroids)

    X =  np.array([[50,224,42],[82,207,277],[22,200,253],[56,285,64],[39,78,256],[62,181,92]])
    k_m = km.KMeans(X, 2, options)
    k_m.centroids =  np.array([[59.0,142.0,30.0],[103.0,248.0,181.0]])
#    np.copyto(k_m.centroids, np.array([[59.0,142.0,30.0],[103.0,248.0,181.0]]))
    k_m._iterate()
    print(k_m.centroids)

    k_m = km.KMeans(X, 2, options)
    k_m.centroids = np.array([[59.0, 142.0, 30.0], [103.0, 248.0, 181.0]])
    #    np.copyto(k_m.centroids, np.array([[59.0,142.0,30.0],[103.0,248.0,181.0]]))
    k_m._cluster_points()
    k_m._get_centroids()
    print(k_m.centroids)

    print(k_m.centroids)
    k_m.run()
    print(k_m.centroids)


#    X = np.reshape(im, (-1, im.shape[2]))
#    print(X)
#     results = []
#     for k in range(1,13):
#         plt.figure(3)
#         options = {'verbose':True, 'km_init':'first'}
#
#         k_m = km.KMeans(X, k, options)
#         t = time.time()
#         k_m.run()
#         print(time.time()-t)
#         results.append(k_m.fitting())
#         plt.figure(2)
#         plt.cla()
#         plt.plot(range(1,k+1),results)
#         plt.xlabel('K')
#         plt.ylabel('fitting score')
#         plt.draw()
#         plt.pause(0.01)
#
#     print k_m.centroids
#
#     plt.figure(3)
#     k_m.plot()
# #plt.savefig('foo.png', bbox_inches='tight')