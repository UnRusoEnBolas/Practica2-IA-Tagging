# -*- coding: utf-8 -*-
"""

@author: Ana Aguilera, Jordi Giménez, Agustín Molina
"""
import re
import numpy as np
import ColorNaming as cn
from skimage import color
import KMeans as km

def NIUs():
    return 1496128, 1425098, 1493035

def loadGT(fileName):
    """@brief   Loads the file with groundtruth content

    @param  fileName  STRING    name of the file with groundtruth

    @return groundTruth LIST    list of tuples of ground truth data
                                (Name, [list-of-labels])
    """

    groundTruth = []
    fd = open(fileName, 'r')
    for line in fd:
        splitLine = line.split(' ')[:-1]
        labels = [''.join(sorted(filter(None,re.split('([A-Z][^A-Z]*)',l)))) for l in splitLine[1:]]
        groundTruth.append( (splitLine[0], labels) )

    return groundTruth


def evaluate(description, GT, options):
    """@brief   EVALUATION FUNCTION
    @param description LIST of color name lists: contain one list of color labels for every images tested
    @param GT LIST images to test and the real color names (see  loadGT)
    @options DICT  contains options to control metric, ...
    @return mean_score,scores mean_score FLOAT is the mean of the scores of each image
                              scores     LIST contain the similiraty between the ground truth list of color names and the obtained
    """
    score = np.array([])

    for i in range(len(description)):
        score = np.append(score, similarityMetric(description[i], GT[i][1], options))

    return  np.mean(score), score



def similarityMetric(Est, GT, options):
    """@brief   SIMILARITY METRIC
    @param Est LIST  list of color names estimated from the image ['red','green',..]
    @param GT LIST list of color names from the ground truth
    @param options DICT  contains options to control metric, ...
    @return S float similarity between label LISTs
    """
    similarity = 0

    if options['metric'].lower() == 'basic'.lower():
        for c in Est:
            if c in GT:
                similarity += 1
        similarity = similarity / len(Est)

    return similarity

def getLabels(kmeans, options):
    """@brief   Labels all centroids of kmeans object to their color names

    @param  kmeans  KMeans      object of the class KMeans
    @param  options DICTIONARY  options necessary for labeling

    @return colors  LIST    colors labels of centroids of kmeans object
    @return ind     LIST    indexes of centroids with the same color label
    """

    if kmeans.centroids.shape[1]<11:
        cent = cn.ImColorNamingTSELabDescriptor(kmeans.centroids)
    else:
        cent = kmeans.centroids
    meaningful_colors = []
    unique = []

    for c in cent:
        bool_array = c>options['single_thr']
        if bool_array[bool_array==True].shape[0]:

            max_arg = np.argmax(c)
            color_to_append = cn.colors[max_arg]
            arg_to_append = max_arg
        else:
            sorted_idx = np.flip(np.argsort(c), axis = 0)

            color_to_append = np.sort([cn.colors[sorted_idx[0]], cn.colors[sorted_idx[1]]])
            color_to_append = color_to_append[0]+color_to_append[1]
            if cn.colors[sorted_idx[0]] > cn.colors[sorted_idx[1]]:
                arg_to_append = [sorted_idx[0], sorted_idx[1]]
            else:
                arg_to_append = [sorted_idx[1], sorted_idx[0]]

        if color_to_append not in meaningful_colors:
            meaningful_colors.append(color_to_append)
            unique.append(arg_to_append)
    print(meaningful_colors, unique)
    return meaningful_colors, unique

def processImage(im, options):
    """@brief   Finds the colors present on the input image

    @param  im      LIST    input image
    @param  options DICTIONARY  dictionary with options

    @return colors  LIST    colors of centroids of kmeans object
    @return indexes LIST    indexes of centroids with the same label
    @return kmeans  KMeans  object of the class KMeans
    """

#########################################################
##  YOU MUST ADAPT THE CODE IN THIS FUNCTIONS TO:
##  1- CHANGE THE IMAGE TO THE CORRESPONDING COLOR SPACE FOR KMEANS
##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
##  3- GET THE NAME LABELS DETECTED ON THE 11 DIMENSIONAL SPACE
#########################################################
    if options['colorspace'].lower() == 'HSV'.lower():
        im = color.rgb2hsv(im)

##  2- APPLY KMEANS ACCORDING TO 'OPTIONS' PARAMETER
    if options['K']<2: # find the bes K
        kmeans = km.KMeans(im, 0, options)
        kmeans.bestK()
    else:
        kmeans = km.KMeans(im, options['K'], options)
        kmeans.run()

    if options['colorspace'].lower() == 'HSV'.lower():
        im = color.rgb2hsv(im)
        kmeans.centroids = color.hsv2rgb(np.array([[kmeans.centroids]]))*255
        print(kmeans.centroids)

#########################################################
##  THE FOLLOWING 2 END LINES SHOULD BE KEPT UNMODIFIED
#########################################################
    colors, which = getLabels(kmeans, options)
    return colors, which, kmeans