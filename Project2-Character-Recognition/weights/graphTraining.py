import sys
import os
import re
import csv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import pdb

fileNames = [   "52-156-trainrecord-run1.csv"]
                #"52-156-trainrecord-run2.csv",
                #"52-156-trainrecord-run3.csv"]

def readCsv(fileName):
    results = []
    with open(fileName, "r") as fp:
        csvReader = csv.reader(fp, delimiter=",")
        for row in csvReader:
            results.append([int(row[0]), float(row[1])])

    array = np.array(results).T
    


    return np.array(results).T

def readAllNames(fns):
    result = None
    for fileName in fns:
        if result is None:
            result = readCsv(fileName)
        else:
            result = np.append(result, readCsv(fileName), axis=1)
    return result

myarr = readAllNames(fileNames)

def makeGraphs(resultSets, title):
    """
    Displays the resultant data sets, along with a given title
    """
    fig, ax = plt.subplots(1)

    ax.plot(resultSets[0][1:100], resultSets[1][1:100])

    ax.legend()
    plt.xlabel("Iteration Number")
    plt.ylabel("Error (squared), per image")
    plt.yscale("log")

    fig.suptitle(title)
    fig.set_size_inches(10,6)

    plt.show() #uncomment this to display the graph on your screen

changeIndices = np.where(myarr[1][:-1] != myarr[1][1:])[0]
xSeries = myarr[0][changeIndices]
ySeries = myarr[1][changeIndices]
myarr = np.stack((xSeries, ySeries))

makeGraphs(myarr, "Error across Training")
