# Imports
''' general: '''
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

''' for entropy: '''
from sklearn.neighbors import KernelDensity

''' for skewness & kurtosis: '''
from scipy.stats import skew
from scipy.stats import kurtosis

''' for fourier transform '''
from scipy import signal as scipysig

''' for regression features'''
import statsmodels.api as smaller

# =============================================================================================================================

# Classification of patientFiles & controlFiles

'''
    Parameter: None
    Returns: list of controlFiles (their file paths), list of patientFiles (their file paths)
'''

def fileClassification(): 
    controlFiles = []
    for i in range(1, 16): 
        if i < 10: 
            controlFiles.append('/work/ParkinsonHW/C_000' + str(i) + '.txt')
        else: 
            controlFiles.append('/work/ParkinsonHW/C_00' + str(i) + '.txt')

    patientFiles = []
    directory = os.fsencode('/work/ParkinsonHW')
    for file in os.listdir(directory): 
        filename = os.fsdecode(file)
        if filename[0] == 'P' or filename[0] == 'H': 
            path = os.path.join('/work/ParkinsonHW', filename)
            patientFiles.append(path)
    
    return controlFiles, patientFiles

# =============================================================================================================================

# Smooth Features

'''
    Parameters: curve (curve = np.array([x, y]), where x and y are the x & y coordinates of drawing), n (desired number of data points)
    smoothing_factor (currently set to 10000 (?))
'''

def smoothCurveFeature(curve, n, smoothing_factor):
    sx = interpolate.UnivariateSpline(np.arange(curve.shape[1]), curve[0,:], k=4)
    sy = interpolate.UnivariateSpline(np.arange(curve.shape[1]), curve[1,:], k=4)
    pressure_f = interpolate.UnivariateSpline(np.arange(np.shape(df[3])[0]), np.array(df[3]), k=4)

    sx.set_smoothing_factor(smoothing_factor)
    sy.set_smoothing_factor(smoothing_factor)
    pressure_f.set_smoothing_factor(smoothing_factor)

    sxdot = sx.derivative()
    sydot = sy.derivative()
    
    sxdotdot = sxdot.derivative()
    sydotdot = sydot.derivative()

    sxdotdotdot = sxdotdot.derivative()
    sydotdotdot = sydotdot.derivative()
    
    t = np.linspace(0, curve.shape[1], n)
    new_curve = np.zeros((2, n))
    new_curve[0,:] = sx(t)
    new_curve[1,:] = sy(t)

    #calculate velocity
    velocity = np.sqrt((sydot(t))**2 + (sxdot(t))**2)

    #calculate acceleration
    acceleration = np.sqrt((sydotdot(t))**2 + (sxdotdot(t))**2)

    #calculate jerk
    jerk = np.sqrt((sydotdotdot(t))**2 + (sxdotdotdot(t))**2)
    
    # calculate curvature
    curvature = (sxdot(t) * sydotdot(t) - sydot(t) * sxdotdot(t))/(sxdot(t)**2 + sydot(t)**2)**(3/2)

    pressure = pressure_f(t)
    
    # new_curve: interpolated/transformed curve, curv_spline_eval: curvature, curv_dot_eval: rate of change of curvature
    return velocity, acceleration, jerk, curvature, pressure

# =============================================================================================================================

# Split static & dynamic drawings
''' 
    Parameter: df (overall pandas dataframe)
    Return: static_df (dataframe containing only static drawings) & dynamic_df (dataframe containing only dynamic drawings)
'''
def staticDynamicSplit(df): 
    if 0 in df[6].unique(): 
        static_df = df[df[6]==0]
    else: 
        static_df = pd.DataFrame()

    if 1 in df[6].unique(): 
        dynamic_df = df[df[6]==1].reset_index()
    else: 
        dynamic_df = pd.DataFrame()

    return static_df, dynamic_df

# =============================================================================================================================

# Entropy
'''
    Parameter: Pandas series/numpy array of desired data
    Return: entropy
'''
def entropyCalc(data): 
    kde = KernelDensity(kernel='gaussian', bandwidth=10).fit(data[:, None])
    logprob = kde.score_samples(data[:, None])
    prob = np.exp(logprob)

    entropy = -1 * sum([i * math.log(i, 2.0) for i in prob])
    return entropy
    
# =============================================================================================================================

# Separating pressure into rising edge, main signal and falling edge

'''
    Parameter: Pandas series/numpy array of pressure data
    Return: index where rising edge ends, index where falling edge starts
'''

def mainSignalThreshold(pressure): 
    risingIndex = 0
    fallingIndex = 0
    rising_threshold = 1.01
    falling_threshold = 0.7 # smaller threshold means steeper drop
    for i in range(1, (int)(len(pressure)/4)): 
        differential = pressure[i+15]/pressure[i]
        if differential <= rising_threshold and pressure[i+20]-pressure[0]>200 and not pressure[i+15]< pressure[i]: 
            risingIndex = i
            break
    for i in range(len(pressure) - (int)(len(pressure)/4), len(pressure)-5): #scan through last quarter
        differential = pressure[i+5]/pressure[i]
        if differential <= falling_threshold: 
            fallingIndex = i
            break
    if fallingIndex == 0:
        fallingIndex = len(pressure)-1

    return risingIndex, fallingIndex

# =============================================================================================================================

# Kinematic Features (can be applied to velocity, acceleration, jerk)

'''
    Use np.mean, np.max, np.median, np.std for basic kinematic features
'''

'''
    rate of inversions in data
    Parameters: data - Pandas series/numpy array of desired data; time - Pandas series/numpy array of time stamp points
    Returns: a numeric value of the number of inversions
'''

def rateOfInversions(data, time):
    maximum = argrelextrema(np.array(data), np.greater)    
    return len(maximum[0]) / (max(time) - min(time))

# =============================================================================================================================

# Fourier Transform 

''' 
    The following 4 functions are helper functions for the fourierFreqCalc function 
'''
def butter_lowpass_filter(data, cut, fs, order=5):
    b, a = scipysig.butter(order, cutoff, btype='low', analog=False)
    y = scipysig.filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cut, fs, order=5):
    b, a = scipysig.butter(order, cutoff, btype='high', analog=False)
    y = scipysig.filtfilt(b, a, data)
    return y

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipysig.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipysig.lfilter(b, a, data)
    return y

'''
    Parameter: data - desired data (Pandas series or Numpy array); time - time series data (Pandas series or Numpy array) 
    (df[5] in the case of ParkinsonHW)
    Returns: list of low frequency metric and list of high frequency metric
'''
def fourierFreqCalc(data, time): 
    cutoff = 0.12
    high_frequency_metric = []
    low_frequency_metric = []
    data = data - np.mean(data)

    # time step
    N = len(time)
    dt = time[1] - time[0]

    # apply lowpass filter
    data_lowpass_filtered = butter_lowpass_filter(data, cutoff, dt)
    data_highpass_filtered = butter_highpass_filter(data, cutoff, dt)

    # calculate
    high_frequency_metric = np.linalg.norm((data_lowpass_filtered - data)/np.linalg.norm(data))
    low_frequency_metric = np.linalg.norm((data_highpass_filtered - data)/np.linalg.norm(data))

    return low_frequency_metric, high_frequency_metric

# =============================================================================================================================

# Regression (velocity, curvature & pressure)

'''
    Parmaeter: NumPy array or Pandas series of data (velocity, curvature, or pressure) (use main signal for pressure)
    Returns: R-squared value, X0 constant, X1 constant and sum of residuals of regression
    Note: curvature - smoothing factor of 100000; velocity - smoothing factor of 10000
'''
def regression(data): 
    model = sm.OLS(pressure, sm.add_constant(np.array(range(len(pressure)))))
    results = model.fit()
    return results.rsquared, results.params[0], results.params[1], sum(abs(results.resid))

# =============================================================================================================================

# Duration & Range of Pressure Rising & Falling Edges

'''
    risingIndex, fallingIndex are from the mainSignalThreshold function

    Rising edge duration: df[5][risingIndex] - df[5][0]
    Falling edge duration: df[5][len(df[5])-1] - df[5][fallingIndex]

    Rising edge range: df[3][risingIndex] - df[3][0] (range of pressure)
    falling_range = df[3][fallingIndex] - df[3][len(df[5])-1] (range of pressure)
'''