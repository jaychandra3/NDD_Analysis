# Imports
''' general: '''
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import interpolate
import scipy
import statistics
from scipy.signal import argrelextrema

''' for entropy: '''
from sklearn.neighbors import KernelDensity

''' for skewness & kurtosis: '''
from scipy.stats import skew
from scipy.stats import kurtosis

''' for fourier transform '''
from scipy import signal as scipysig

''' for regression features'''
import statsmodels.api as sm

# =============================================================================================================================

# Normal velocity variability

'''
    Parameters: NumPy array/Pandas Series of desired velocity
    Returns: normal velocity variability (double)
'''
def nvv(data, timestamps): 
    sigma_sum = 0
    for i in range(1, len(data) - 1): 
        sigma_sum += abs(data[i+1] - data[i])
    T = max(timestamps) - min(timestamps)
    nvv = 1/(T * abs(np.mean(data))) * sigma_sum
    return nvv

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
    smoothing_factor (currently set to 10000, right?)
    Returns: velocity, acceleration, jerk, curvature, pressure
'''

def smoothCurveFeature(curve, n, smoothing_factor, df):
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


'''
    Parameters: input R - NumPy array/Pandas series of spiral radius; inputT - NumPy array/Pandas series of theta at each point
    along the spiral; n - desired number of data points to truncate to; smoothing_factor - currently set to 1000
    Returns: radius - array of smoothed radius; theta - array of smoothed theta; velocityR (r'(t) - derivative of radius as a function of time); 
    velocityT (theta'(t)); accelerationR (r''(t)); accelerationT (r''(theta)), drdtheta
'''

def smoothPolarFeature(inputR, inputT, n, smoothing_factor):
    sx = interpolate.UnivariateSpline(np.arange(len(inputR)), inputR, k=4)
    sy = interpolate.UnivariateSpline(np.arange(len(inputT)), inputT, k=4)

    sx.set_smoothing_factor(smoothing_factor)
    sy.set_smoothing_factor(smoothing_factor)

    sxdot = sx.derivative()
    sydot = sy.derivative()
    
    sxdotdot = sxdot.derivative()
    sydotdot = sydot.derivative()
    
    t = np.linspace(0, len(inputR), n)

    radius = sx(t)
    theta = sy(t)

    #calculate velocity
    velocityR = sxdot(t)
    velocityT = sydot(t)

    #calculate acceleration
    accelerationR = sxdotdot(t)
    accelerationT = sydotdot(t)

    #dr/dtheta
    drdtheta = velocityR/velocityT

    # new_curve: interpolated/transformed curve, curv_spline_eval: curvature, curv_dot_eval: rate of change of curvature
    return (radius, theta, velocityR, velocityT, accelerationR, accelerationT, drdtheta)

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
        static_df = pd.DataFrame(columns = df.columns)

    if 1 in df[6].unique(): 
        dynamic_df = df[df[6]==1].reset_index()
    else: 
        dynamic_df = pd.DataFrame(columns = df.columns)

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
    return (len(maximum[0]) / (max(time) - min(time)))

# =============================================================================================================================

# Fourier Transform 

''' 
    The following 4 functions are helper functions for the fourierFreqCalc function 
'''
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = scipysig.butter(order, cutoff, btype='low', analog=False)
    y = scipysig.filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
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
def fourierFreqCalc(data, time, cutoff, low, high): 
    high_frequency_metric = []
    low_frequency_metric = []
    bandpass_frequency_metric = []
    data = data - np.mean(data)

    # time step
    # dt = time[1] - time[0]
    dt = 1/60

    # apply lowpass filter
    data_lowpass_filtered = butter_lowpass_filter(data, cutoff, 1/dt)
    data_highpass_filtered = butter_highpass_filter(data, cutoff, 1/dt)
    data_bandpass_filtered = butter_bandpass_filter(data, low, high, 1/dt)

    # calculate
    high_frequency_metric = np.linalg.norm((data_lowpass_filtered - data)/np.linalg.norm(data))
    low_frequency_metric = np.linalg.norm((data_highpass_filtered - data)/np.linalg.norm(data))
    bandpass_frequency_metric = np.linalg.norm((data_bandpass_filtered - data)/np.linalg.norm(data))

    return low_frequency_metric, high_frequency_metric, bandpass_frequency_metric

# =============================================================================================================================

# Linear Regression (used for velocity, pressure, and more)

'''
    This is for linear regression on some kind of signal versus time
    Parmaeter: 
        data - NumPy array or Pandas series of data (velocity/pressure main signal); 
    Returns: R-squared value, X0 constant, X1 constant and sum of residuals of regression
    Note: velocity - smoothing factor of 10000; pressure - original data; no smoothing
'''
def time_regression(data):
    model = sm.OLS(data, sm.add_constant(np.array(range(len(data)))))
    results = model.fit()
    return results.rsquared, results.params[0], results.params[1], sum(abs(results.resid))

'''
    This is for linear regression on a signal versus another signal
    Parameter: 
        data1 & data - NumPy array or Pandas series of data (data1 - dependent variable; data2 - independent variable)
    Returns: R-squared value, X0 constant, X1 constant and sum of residuals of regression
'''
def nontime_regression(data1, data2): 
    model = sm.OLS(data1, sm.add_constant(data2))
    results = model.fit()
    return results.rsquared, results.params[0], results.params[1], sum(abs(results.resid))

# Nonlinear Regression (used for curvature)

'''
    Nonlinear regression on some kind of signal vs. time
    Parameter: 
        data - NumPy array or Pandas series of data (curvature)
        function - function to model the regression after (func_log for log regression, func_inv for regression that fits inversely 
        proportional relationships)
    Returns: R-squared value, X0 constant, X1 constant and sum of residuals of the particular type of regression
    Note: curvature - smoothing factor of 100000
'''
def nonlinear_time_regression(data, function): 
    xdata = np.array(range(len(data))) + 1
    param, cov = scipy.optimize.curve_fit(function, xdata, data)
    residuals = data - function(xdata, *param)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data - np.mean(data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared, param[0], param[1], ss_res

'''
    Nonlinear time regression on a signal versus another signal
    Parameters: 
        data1 - the dependent variable; data2 - the independent variable 
    Returns: 
        R-squared value, X0 constant, X1 constant and sum of residuals of the particular type of regression
'''
def nonlinear_nontime_regression(data1, data2, function): 
    param, cov = scipy.optimize.curve_fit(function, data2, data1)
    residuals = data1 - function(data2, *param)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((data1-np.mean(data1))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared, param[0], param[1], ss_res

'''
    Helper function for logarithmic regressions
    Parameter: x - set of x values; a - 
'''
def func_log(x, a, b): 
    return a + b*np.log(x)

'''
    Helper function for regression that fits inversely proportional relationships
'''
def func_inv(x, a, b): 
    return a + b/(x + 10**(-10))

# =============================================================================================================================

# Duration & Range of Pressure Rising & Falling Edges

'''
    risingIndex, fallingIndex are from the mainSignalThreshold function

    Rising edge duration: df[5][risingIndex] - df[5][0]
    Falling edge duration: df[5][len(df[5])-1] - df[5][fallingIndex]

    Rising edge range: df[3][risingIndex] - df[3][0] (range of pressure)
    falling_range = df[3][fallingIndex] - df[3][len(df[5])-1] (range of pressure)
'''

# =============================================================================================================================

''' Static Preprocessing Function '''

def static_preprocessing(filename): 
    df = pd.read_csv(filename, sep = ";", header = None)

    # split into static and dynamic dataframes
    static_df, _ = staticDynamicSplit(df)

    if static_df.count == 0: 
        return [np.nan for i in list(range(0, 14))]

    static_time = list(static_df[5])

    static_x = static_df[0]
    static_y = static_df[1]
    static_curve = np.array([static_x, static_y])

    # radius & theta calculations
    static_x0 = float(static_x[0])
    static_y0 = float(static_y[0])
    static_r = np.array(((static_x - static_x0)**2 + (static_y - static_y0)**2)**(1/2))
    static_t0 = np.arctan((static_y - static_y0) / (static_x - static_x0+(10**(-10))))
    static_t = np.cumsum(np.abs(np.diff(np.abs(static_t0))))

    static_velocity, static_acceleration, static_jerk, _, _ = smoothCurveFeature(static_curve, 1000, 10000, static_df)
    static_velocity = static_velocity[50: len(static_velocity)-50]
    static_acceleration = static_acceleration[50: len(static_acceleration)-50]
    static_jerk = static_jerk[50: len(static_jerk)-50]

    # radius & theta rate of change features
    static_radius, static_theta, static_rdot, static_tdot, static_rdotdot, static_tdotdot, static_drdtheta = smoothPolarFeature(static_r, static_t, 1000, 1000)
    static_radius = static_radius[50:len(static_radius)-50]
    static_theta = static_theta[50:len(static_theta)-50]
    
    _, _, _, static_curvature, _ = smoothCurveFeature(static_curve, 1000, 100000, static_df)
    static_curvature = static_curvature[50: len(static_curvature)-50] # got rid of last 100 before
    
    static_pressure = static_df[3]
    static_risingIndex, static_fallingIndex = mainSignalThreshold(static_pressure)

    static_pressure_rising = static_pressure[0:static_risingIndex]
    static_pressure_main = static_pressure[static_risingIndex:static_fallingIndex]
    static_pressure_falling = static_pressure[static_fallingIndex:-1]

    static_altitude = static_df[4]

    return (static_time, static_x, static_y, static_radius, static_theta, static_velocity, static_acceleration, 
    static_jerk, static_rdot, static_tdot, static_rdotdot, static_tdotdot, static_drdtheta, static_curvature, 
    static_pressure, static_risingIndex, static_fallingIndex, static_pressure_rising, static_pressure_main, 
    static_pressure_falling, static_altitude)

''' Dynamic Preprocessing Function '''

def dynamic_preprocessing(filename):
    df = pd.read_csv(filename, sep = ";", header = None)

    # split into static and dynamic dataframes
    _, dynamic_df = staticDynamicSplit(df)

    if dynamic_df.count == 0:
        return [np.nan for i in list(range(0, 14))]

    dynamic_time = list(dynamic_df[5])

    dynamic_x = dynamic_df[0]
    dynamic_y = dynamic_df[1]
    dynamic_curve = np.array([dynamic_x, dynamic_y])

    # radius & theta calculations
    dynamic_x0 = float(dynamic_x[0])
    dynamic_y0 = float(dynamic_y[0])
    dynamic_r = np.array(((dynamic_x - dynamic_x0)**2 + (dynamic_y - dynamic_y0)**2)**(1/2))
    dynamic_t0 = np.arctan((dynamic_y - dynamic_y0) / (dynamic_x - dynamic_x0+(10**(-10))))
    dynamic_t = np.cumsum(np.abs(np.diff(np.abs(dynamic_t0))))

    dynamic_velocity, dynamic_acceleration, dynamic_jerk, _, _ = smoothCurveFeature(dynamic_curve, 1000, 10000, dynamic_df)
    dynamic_velocity = dynamic_velocity[50: len(dynamic_velocity)-50]
    dynamic_acceleration = dynamic_acceleration[50: len(dynamic_acceleration)-50]
    dynamic_jerk = dynamic_jerk[50: len(dynamic_jerk)-50]

    # radius & theta rate of change features
    dynamic_radius, dynamic_theta, dynamic_rdot, dynamic_tdot, dynamic_rdotdot, dynamic_tdotdot, dynamic_drdtheta = smoothPolarFeature(dynamic_r, dynamic_t, 1000, 1000)
    dynamic_radius = dynamic_radius[50:len(dynamic_radius)-50]
    dynamic_theta = dynamic_theta[50:len(dynamic_theta)-50]

    _, _, _, dynamic_curvature, _ = smoothCurveFeature(dynamic_curve, 1000, 100000, dynamic_df)
    dynamic_curvature = dynamic_curvature[50: len(dynamic_curvature)-50] # got rid of last 100 before

    dynamic_pressure = dynamic_df[3]
    dynamic_risingIndex, dynamic_fallingIndex = mainSignalThreshold(dynamic_pressure)

    dynamic_pressure_rising = dynamic_pressure[0:dynamic_risingIndex]
    dynamic_pressure_main = dynamic_pressure[dynamic_risingIndex:dynamic_fallingIndex]
    dynamic_pressure_falling = dynamic_pressure[dynamic_fallingIndex:-1]

    dynamic_altitude = dynamic_df[4]

    return (dynamic_time, dynamic_x, dynamic_y, dynamic_radius, dynamic_theta, dynamic_velocity, dynamic_acceleration,
    dynamic_jerk, dynamic_rdot, dynamic_tdot, dynamic_rdotdot, dynamic_tdotdot, dynamic_drdtheta, dynamic_curvature,
    dynamic_pressure, dynamic_risingIndex, dynamic_fallingIndex, dynamic_pressure_rising, dynamic_pressure_main,
    dynamic_pressure_falling, dynamic_altitude)

# =============================================================================================================================

'''Calculates all static features '''

def static_calculate(filename): 
    # entropy:
    static_x_entropy = entropyCalc(static_x)
    static_y_entropy = entropyCalc(static_y)

    # kinematic features
    # velocity features
    static_velocity_mean = np.mean(static_velocity)
    static_velocity_std = np.std(static_velocity)
    static_velocity_max = max(static_velocity)
    static_velocity_inversion_rate = rateOfInversions(static_velocity, static_time)
    static_nvv = nvv(static_velocity, static_time)

    # acceleration features
    static_acceleration_mean = np.mean(static_acceleration)
    static_acceleration_std = np.std(static_acceleration)
    static_acceleration_max = max(static_acceleration)
    static_acceleration_inversion_rate = rateOfInversions(static_acceleration, static_time)

    # jerk features
    static_jerk_mean = np.mean(static_jerk)
    static_jerk_std = np.std(static_jerk)
    static_jerk_max = max(static_jerk)
    static_jerk_inversion_rate = rateOfInversions(static_jerk, static_time)

    # basic pressure features
    static_pressure_mean = np.mean(static_pressure)
    static_pressure_std = np.std(static_pressure)
    static_pressure_max = max(static_pressure)
    static_pressure_inversion_rate = rateOfInversions(static_pressure, static_time)

    # curvature rate of inversion
    static_curv_inversion_rate = rateOfInversions(static_curvature, static_time)

    # skewness & kurtosis
    static_x_skewness = skew(static_x)
    static_y_skewness = skew(static_y)
    static_x_kurtosis = kurtosis(static_x)
    static_y_kurtosis = kurtosis(static_y)
    
    static_vel_skewness = skew(static_velocity)
    static_vel_kurtosis = kurtosis(static_velocity)

    static_accel_skewness = skew(static_acceleration)
    static_accel_kurtosis = kurtosis(static_acceleration)

    static_jerk_skewness = skew(static_jerk)
    static_jerk_kurtosis = kurtosis(static_jerk)

    static_pressure_skewness = skew(static_pressure)
    static_pressure_kurtosis = kurtosis(static_pressure)

    static_curv_skewness = skew(static_curvature)
    static_curv_kurtosis = kurtosis(static_curvature)

    # fourier transform pressure
    static_pressure_low_freq, static_pressure_high_freq, static_pressure_bandpass_freq = fourierFreqCalc(static_pressure_main, static_time, 0.12, 0.3, 0.8)

    # fourier transform altitude
    static_altitude_low_freq, static_altitude_high_freq, static_altitude_bandpass_freq = fourierFreqCalc(static_altitude, static_time, 0.12, 0.25, 0.6)

    # pressure vs. time linear regression fit
    static_pressure_reg_main_r2, static_pressure_reg_main_x0, static_pressure_reg_main_x1, static_pressure_reg_main_sumresid = time_regression(static_pressure_main)

    # curvature vs. time logarithmic regression fit
    static_curv_reg_r2, static_curv_reg_x0, static_curv_reg_x1, static_curv_reg_sumresid = nonlinear_time_regression(static_curvature, func_log)

    # velocity vs. time linear regression fit
    static_velocity_reg_r2, static_velocity_reg_x0, static_velocity_reg_x1, static_velocity_reg_sumresid = time_regression(static_velocity)

    # linear regression fit for velocity vs. radius
    static_VR_reg_r2, static_VR_reg_x0, static_VR_reg_x1, static_VR_reg_sumresid = nontime_regression(static_velocity, static_radius)

    # inversely proportional fit for curvature vs. velocity
    static_CV_reg_r2, static_CV_reg_x0, static_CV_reg_x1, static_CV_reg_sumresid = nonlinear_nontime_regression(static_curvature, static_velocity, func_inv)

    # radius vs. theta linear regression fit 
    static_RT_reg_r2, static_RT_reg_x0, static_RT_reg_x1, static_RT_reg_sumresid = nontime_regression(static_radius, static_theta)

    static_rdot_mean = np.mean(static_rdot)
    static_rdot_std = np.std(static_rdot)
    static_tdot_mean = np.mean(static_tdot)
    static_tdot_std = np.std(static_tdot)
    static_rdotdot_mean = np.mean(static_rdotdot)
    static_rdotdot_std = np.std(static_rdotdot)
    static_tdotdot_mean = np.mean(static_tdotdot)
    static_tdotdot_std = np.std(static_tdotdot)
    static_drdtheta_mean = np.mean(static_drdtheta)
    static_drdtheta_std = np.std(static_drdtheta)

    # pressure rising & falling duration/range
    static_pressure_rising_duration = static_time[static_risingIndex] - static_time[0]
    static_pressure_rising_range = static_pressure[static_risingIndex] - static_pressure[0]
    static_pressure_falling_duration = static_time[len(static_time)-1] - static_time[static_fallingIndex]
    static_pressure_falling_range = static_pressure[static_fallingIndex] - static_pressure[len(static_time)-1]

    # overall duration
    static_duration = static_time[-1] - static_time[0]

    return (static_velocity_mean, static_velocity_max, static_velocity_std, static_nvv, static_velocity_inversion_rate,
    static_acceleration_mean, static_acceleration_max, static_acceleration_std, static_acceleration_inversion_rate,
    static_jerk_mean, static_jerk_max, static_jerk_std, static_jerk_inversion_rate,
    static_duration, static_curv_inversion_rate,
    static_pressure_mean, static_pressure_max, static_pressure_std, static_pressure_inversion_rate,
    static_x_entropy, static_y_entropy, static_x_skewness, static_y_skewness, static_x_kurtosis, static_y_kurtosis,
    static_vel_skewness, static_vel_kurtosis, static_accel_skewness, static_accel_kurtosis, static_jerk_skewness, static_jerk_kurtosis,
    static_pressure_skewness, static_pressure_kurtosis, static_curv_skewness, static_curv_kurtosis,
    static_pressure_high_freq, static_pressure_low_freq, static_pressure_bandpass_freq, 
    static_altitude_high_freq, static_altitude_low_freq, static_altitude_bandpass_freq,
    static_pressure_reg_main_r2, static_pressure_reg_main_x0, static_pressure_reg_main_x1, static_pressure_reg_main_sumresid,
    static_curv_reg_r2, static_curv_reg_x0, static_curv_reg_x1, static_curv_reg_sumresid,
    static_velocity_reg_r2, static_velocity_reg_x0, static_velocity_reg_x1, static_velocity_reg_sumresid,
    static_VR_reg_r2, static_VR_reg_x0, static_VR_reg_x1, static_VR_reg_sumresid, 
    static_CV_reg_r2, static_CV_reg_x0, static_CV_reg_x1, static_CV_reg_sumresid, 
    static_RT_reg_r2, static_RT_reg_x0, static_RT_reg_x1, static_RT_reg_sumresid, 
    static_rdot_mean, static_rdot_std, static_tdot_mean, static_tdot_std, 
    static_rdotdot_mean, static_rdotdot_std, static_tdotdot_mean, static_tdotdot_std, 
    static_drdtheta_mean, static_drdtheta_std, 
    static_pressure_rising_duration, static_pressure_rising_range, static_pressure_falling_duration, static_pressure_falling_range)

''' Calculate all dynamic features '''

'''
Parameter: filename - file path of patient csv
Returns: dynamic features
'''

def dynamic_calculate(filename):
    # entropy:
    dynamic_x_entropy = entropyCalc(dynamic_x)
    dynamic_y_entropy = entropyCalc(dynamic_y)

    # kinematic features
    # velocity features
    dynamic_velocity_mean = np.mean(dynamic_velocity)
    dynamic_velocity_std = np.std(dynamic_velocity)
    dynamic_velocity_max = max(dynamic_velocity)
    dynamic_velocity_inversion_rate = rateOfInversions(dynamic_velocity, dynamic_time)
    dynamic_nvv = nvv(dynamic_velocity, dynamic_time)

    # acceleration features
    dynamic_acceleration_mean = np.mean(dynamic_acceleration)
    dynamic_acceleration_std = np.std(dynamic_acceleration)
    dynamic_acceleration_max = max(dynamic_acceleration)
    dynamic_acceleration_inversion_rate = rateOfInversions(dynamic_acceleration, dynamic_time)

    # jerk features
    dynamic_jerk_mean = np.mean(dynamic_jerk)
    dynamic_jerk_std = np.std(dynamic_jerk)
    dynamic_jerk_max = max(dynamic_jerk)
    dynamic_jerk_inversion_rate = rateOfInversions(dynamic_jerk, dynamic_time)

    # basic pressure features
    dynamic_pressure_mean = np.mean(dynamic_pressure)
    dynamic_pressure_std = np.std(dynamic_pressure)
    dynamic_pressure_max = max(dynamic_pressure)
    dynamic_pressure_inversion_rate = rateOfInversions(dynamic_pressure, dynamic_time)

    # curvature rate of inversion
    dynamic_curv_inversion_rate = rateOfInversions(dynamic_curvature, dynamic_time)

    # skewness & kurtosis
    dynamic_x_skewness = skew(dynamic_x)
    dynamic_y_skewness = skew(dynamic_y)
    dynamic_x_kurtosis = kurtosis(dynamic_x)
    dynamic_y_kurtosis = kurtosis(dynamic_y)

    dynamic_vel_skewness = skew(dynamic_velocity)
    dynamic_vel_kurtosis = kurtosis(dynamic_velocity)

    dynamic_accel_skewness = skew(dynamic_acceleration)
    dynamic_accel_kurtosis = kurtosis(dynamic_acceleration)

    dynamic_jerk_skewness = skew(dynamic_jerk)
    dynamic_jerk_kurtosis = kurtosis(dynamic_jerk)

    dynamic_pressure_skewness = skew(dynamic_pressure)
    dynamic_pressure_kurtosis = kurtosis(dynamic_pressure)

    dynamic_curv_skewness = skew(dynamic_curvature)
    dynamic_curv_kurtosis = kurtosis(dynamic_curvature)

    # fourier transform pressure
    dynamic_pressure_low_freq, dynamic_pressure_high_freq, dynamic_pressure_bandpass_freq = fourierFreqCalc(dynamic_pressure_main, dynamic_time, 0.12, 0.3, 0.8)

    # fourier transform altitude
    dynamic_altitude_low_freq, dynamic_altitude_high_freq, dynamic_altitude_bandpass_freq = fourierFreqCalc(dynamic_altitude, dynamic_time, 0.12, 0.25, 0.6)

    # pressure vs. time linear regression fit
    dynamic_pressure_reg_main_r2, dynamic_pressure_reg_main_x0, dynamic_pressure_reg_main_x1, dynamic_pressure_reg_main_sumresid = time_regression(dynamic_pressure_main)

    # curvature vs. time logarithmic regression fit
    dynamic_curv_reg_r2, dynamic_curv_reg_x0, dynamic_curv_reg_x1, dynamic_curv_reg_sumresid = nonlinear_time_regression(dynamic_curvature, func_log)

    # velocity vs. time linear regression fit
    dynamic_velocity_reg_r2, dynamic_velocity_reg_x0, dynamic_velocity_reg_x1, dynamic_velocity_reg_sumresid = time_regression(dynamic_velocity)

    # linear regression fit for velocity vs. radius
    dynamic_VR_reg_r2, dynamic_VR_reg_x0, dynamic_VR_reg_x1, dynamic_VR_reg_sumresid = nontime_regression(dynamic_velocity, dynamic_radius)

    # inversely proportional fit for curvature vs. velocity
    dynamic_CV_reg_r2, dynamic_CV_reg_x0, dynamic_CV_reg_x1, dynamic_CV_reg_sumresid = nonlinear_nontime_regression(dynamic_curvature, dynamic_velocity, func_inv)

    # radius vs. theta linear regression fit
    dynamic_RT_reg_r2, dynamic_RT_reg_x0, dynamic_RT_reg_x1, dynamic_RT_reg_sumresid = nontime_regression(dynamic_radius, dynamic_theta)

    dynamic_rdot_mean = np.mean(dynamic_rdot)
    dynamic_rdot_std = np.std(dynamic_rdot)
    dynamic_tdot_mean = np.mean(dynamic_tdot)
    dynamic_tdot_std = np.std(dynamic_tdot)
    dynamic_rdotdot_mean = np.mean(dynamic_rdotdot)
    dynamic_rdotdot_std = np.std(dynamic_rdotdot)
    dynamic_tdotdot_mean = np.mean(dynamic_tdotdot)
    dynamic_tdotdot_std = np.std(dynamic_tdotdot)
    dynamic_drdtheta_mean = np.mean(dynamic_drdtheta)
    dynamic_drdtheta_std = np.std(dynamic_drdtheta)

    # pressure rising & falling duration/range
    dynamic_pressure_rising_duration = dynamic_time[dynamic_risingIndex] - dynamic_time[0]
    dynamic_pressure_rising_range = dynamic_pressure[dynamic_risingIndex] - dynamic_pressure[0]
    dynamic_pressure_falling_duration = dynamic_time[len(dynamic_time)-1] - dynamic_time[dynamic_fallingIndex]
    dynamic_pressure_falling_range = dynamic_pressure[dynamic_fallingIndex] - dynamic_pressure[len(dynamic_time)-1]

    # overall duration
    dynamic_duration = dynamic_time[-1] - dynamic_time[0]

    return (dynamic_velocity_mean, dynamic_velocity_max, dynamic_velocity_std, dynamic_nvv, dynamic_velocity_inversion_rate,
    dynamic_acceleration_mean, dynamic_acceleration_max, dynamic_acceleration_std, dynamic_acceleration_inversion_rate,
    dynamic_jerk_mean, dynamic_jerk_max, dynamic_jerk_std, dynamic_jerk_inversion_rate,
    dynamic_duration, dynamic_curv_inversion_rate,
    dynamic_pressure_mean, dynamic_pressure_max, dynamic_pressure_std, dynamic_pressure_inversion_rate,
    dynamic_x_entropy, dynamic_y_entropy, dynamic_x_skewness, dynamic_y_skewness, dynamic_x_kurtosis, dynamic_y_kurtosis,
    dynamic_vel_skewness, dynamic_vel_kurtosis, dynamic_accel_skewness, dynamic_accel_kurtosis, dynamic_jerk_skewness, dynamic_jerk_kurtosis,
    dynamic_pressure_skewness, dynamic_pressure_kurtosis, dynamic_curv_skewness, dynamic_curv_kurtosis,
    dynamic_pressure_high_freq, dynamic_pressure_low_freq, dynamic_pressure_bandpass_freq,
    dynamic_altitude_high_freq, dynamic_altitude_low_freq, dynamic_altitude_bandpass_freq,
    dynamic_pressure_reg_main_r2, dynamic_pressure_reg_main_x0, dynamic_pressure_reg_main_x1, dynamic_pressure_reg_main_sumresid,
    dynamic_curv_reg_r2, dynamic_curv_reg_x0, dynamic_curv_reg_x1, dynamic_curv_reg_sumresid,
    dynamic_velocity_reg_r2, dynamic_velocity_reg_x0, dynamic_velocity_reg_x1, dynamic_velocity_reg_sumresid,
    dynamic_VR_reg_r2, dynamic_VR_reg_x0, dynamic_VR_reg_x1, dynamic_VR_reg_sumresid,
    dynamic_CV_reg_r2, dynamic_CV_reg_x0, dynamic_CV_reg_x1, dynamic_CV_reg_sumresid,
    dynamic_RT_reg_r2, dynamic_RT_reg_x0, dynamic_RT_reg_x1, dynamic_RT_reg_sumresid,
    dynamic_rdot_mean, dynamic_rdot_std, dynamic_tdot_mean, dynamic_tdot_std,
    dynamic_rdotdot_mean, dynamic_rdotdot_std, dynamic_tdotdot_mean, dynamic_tdotdot_std,
    dynamic_drdtheta_mean, dynamic_drdtheta_std,
    dynamic_pressure_rising_duration, dynamic_pressure_rising_range, dynamic_pressure_falling_duration, dynamic_pressure_falling_range)

# =============================================================================================================================

# Feature Selection Functions

''' Run the feature selection functions on the dataframes read in from the CSVs (Static_HW_features.csv & Dynamic_HW_features.csv) '''
 # (include features we want to include, and whether static or dynamic)