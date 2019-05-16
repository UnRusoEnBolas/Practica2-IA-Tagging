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
    options = {'colorspace':'ColorNaming', 'K':3, 'synonyms':False, 'single_thr':0.2, 'verbose':True, 'km_init':'random', 'metric':'basic'}

    ImageFolder = 'Images'
    GTFile = 'LABELSlarge.txt'
    
    GTFile = ImageFolder + '/' + GTFile
    GT = lb.loadGT(GTFile)

    DBcolors = []
    for gt in GT:
        print(gt[0])
        im = io.imread(ImageFolder+"/"+gt[0])    
        colors,_,_ = lb.processImage(im, options)
        DBcolors.append(colors)
        
    encert,_ = lb.evaluate(DBcolors, GT, options)
    print("Encert promig: "+ '%.2f' % (encert*100) + '%')
    print(time.time()-t)