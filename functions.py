import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import scipy as sc
import pandas as pd

#konstanter
g = 9.81
y0 = 0.36689650933919327
c=2/5





trial_2 = np.genfromtxt("data\\trial_2.csv", delimiter=",", skip_header=2)
trial_3 = np.genfromtxt("data\\trial_3.csv", delimiter=",", skip_header=2)
# trial_4 = np.genfromtxt("data\\trial_4.csv", delimiter=",", skip_header=2)
trial_5 = np.genfromtxt("data\\trial_5.csv", delimiter=",", skip_header=2)
# trial_6 = np.genfromtxt("data\\trial_6.csv", delimiter=",", skip_header=2)
# trial_7 = np.genfromtxt("data\\trial_7.csv", delimiter=",", skip_header=2)
trial_8 = np.genfromtxt("data\\trial_8.csv", delimiter=",", skip_header=2)
trial_9 = np.genfromtxt("data\\trial_9.csv", delimiter=",", skip_header=2)
trial_10 = np.genfromtxt("data\\trial_10.csv", delimiter=",", skip_header=2)
trial_11 = np.genfromtxt("data\\trial_11.csv", delimiter=",", skip_header=2)

trials = {
    "trial_2": trial_2,
    "trial_3": trial_3,
    # "trial_4": trial_4,
    "trial_5": trial_5,
    # "trial_6": trial_6,
    # "trial_7": trial_7,
    "trial_8": trial_8,
    "trial_9": trial_9,
    "trial_10": trial_10,
    "trial_11": trial_11,
}




# def extract_averages(columnindex):
#     t2 = extract_column(trial_2, columnindex)
#     t3 = extract_column(trial_3, columnindex)
#     t5 = extract_column(trial_5, columnindex)
#     averages = []
#     for index in range(len(t2)):
#         average = (t2[index] + t3[index] + t5[index])/3
#         averages.append(average)
#     for index in range(len(ex))
#     return np.asarray(averages)

def extract_averages(columnindex):
    averages = []
    first = True
    counter = 0
    for trial in trials.values():
        values = extract_column(trial, columnindex)
        counter += 1
        if first == True:                 # fyller med 0'er så man kan aksessere etter index
            for index in range(len(values)+3):
                averages.append(0)
            first = False
        for index in range(len(values)):
            averages[index] += values[index]
    for index in range(len(averages)):
        averages[index] = averages[index]/counter
    return np.asarray(averages)

def extract_column(trial, columnindex):
    y = []
    for list in trial:
        y.append(list[columnindex])
    return np.asarray(y)


def v(trial):
    y = extract_column(trial, 2)
    return np.sqrt((10 * g * (y0 - y)) / 7)




print(extract_averages(2))


