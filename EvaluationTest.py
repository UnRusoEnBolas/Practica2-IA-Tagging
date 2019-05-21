# -*- coding: utf-8 -*-
"""

@author: ramon
"""
from skimage import io
import matplotlib.pyplot as plt
import os
import time 

if os.path.isfile('TeachersLabels.py') and True: 
    import TeachersLabels as lb
else:
    import Labels as lb



plt.close("all")
if __name__ == "__main__":

    #'colorspace': 'RGB', 'Lab' o 'ColorNaming'
    t = time.time()
    options = {'K':0, 'colorspace':'RGB', 'km_init':'center', 'bestKmethod':'super', 'fitting':'fisher'}




    ImageFolder = 'Images'
    GTFile = 'LABELSlarge.txt'
    
    GTFile = ImageFolder + '/' + GTFile
    GT = lb.loadGT(GTFile)

    DBcolors = []
    kpromedio = 0
    for gt in GT:
        print(gt[0])
        im = io.imread(ImageFolder+"/"+gt[0])
        colors,which, kmeans = lb.processImage(im, options)
        DBcolors.append(colors)
        kpromedio += kmeans.K

    kpromedio = kpromedio/len(GT)

    print("K Promig: ", kpromedio)
    encert,_ = lb.evaluate(DBcolors, GT, options)
    print("Encert promig: "+ '%.2f' % (encert*100) + '%')
    print(time.time()-t)