import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import statistics

# Scalp wavelengths (805 - 2000 nm) and scalp absorption

scalp_absorption = [0.52, 0.40, 0.39, 0.33, 0.19, 0.65,
                    0.50, 1.98, 2.19, 2.04, 1.43, 1.87, 1.73, 2.57, 2.52, 2.09]
scalp_wavelengths = [805, 900, 950, 1000, 1100, 1200, 1300,
                     1400, 1430, 1500, 1600, 1700, 1800, 1900, 1930, 2000]

# Skull wavelengths (801 - 2000 nm) and skull absorption

skull_absorption = [0.11, 0.15, 0.23, 0.22, 0.16, 0.67, 0.67,
                         0.54, 2.43, 3.33, 3.13, 2.47, 2.77, 2.98, 2.97, 4.39, 4.97, 4.47]
skull_wavelengths = [801, 900, 980, 1000, 1100, 1180, 1200,
                          1300, 1400, 1465, 1500, 1600, 1700, 1740, 1800, 1900, 1930, 2000]

# Gray matter wavelengths (400 - 1300 nm) and gray matter absorption

GM_absorption = [9.778, 14.873, 16.722, 5.161, 2.272, 2.206, 2.955, 1.460, 0.925, 0.809, 0.733,
                 0.599, 0.507, 0.485, 0.472, 0.479, 0.503, 0.521, 0.585, 0.502, 0.502, 0.815, 1.010, 0.865, 0.894]
GM_wavelengths = [400, 418, 428, 450, 488, 500, 550, 600, 632, 670, 700, 750,
                  800, 830, 850, 870, 900, 950, 1000, 1064, 1100, 1150, 1200, 1250, 1300]

# White matter wavelengths (400 - 1300 nm) and white matter absorption

WM_absorption = [9.134, 13.603, 15.417, 3.958, 1.869, 1.834, 2.584, 1.175, 0.801, 0.711, 0.674,
                 0.649, 0.622, 0.626, 0.643, 0.666, 0.684, 0.785, 0.883, 0.752, 0.762, 1.135, 1.420, 1.268, 1.274]
WM_wavelengths = [400, 418, 428, 450, 488, 500, 550, 600, 632, 670, 700, 750,
                  800, 830, 850, 870, 900, 950, 1000, 1064, 1100, 1150, 1200, 1250, 1300]

### Interpolation

# Interpolate existing data
scalp_interpolation = interp1d(scalp_wavelengths, scalp_absorption, kind='cubic')
skull_interpolation = interp1d(skull_wavelengths, skull_absorption, kind='cubic')
GM_interpolation = interp1d(GM_wavelengths, GM_absorption, kind='cubic')
WM_interpolation = interp1d(WM_wavelengths, WM_absorption, kind='cubic')

# Generate evenly spaced out numbers within wavelength range
scalp_interp_wv = np.arange(805, 2001, 1)
skull_interp_wv = np.arange(801, 2001, 1)
GM_interp_wv = np.arange(400, 1301, 1)
WM_interp_wv = np.arange(400, 1301, 1)

# Interpolated functions
scalp_lambda = scalp_interpolation(scalp_interp_wv)
skull_lambda = skull_interpolation(skull_interp_wv)
GM_lambda = GM_interpolation(GM_interp_wv)
WM_lambda = WM_interpolation(WM_interp_wv)

GM_scalp_sum = 0
overlap_length = 0

# Avg vertical offset
for x in scalp_interp_wv:
    if (x >= 805) and (x <= 1300):
        GM_scalp_sum += GM_lambda[np.where(GM_interp_wv == x)] - scalp_lambda[np.where(scalp_interp_wv == x)]
        overlap_length += 1

GM_scalp_avg_offset = GM_scalp_sum/(overlap_length)

GM_skull_sum = 0
overlap_length = 0

for y in skull_interp_wv:
    if (y >= 801) and (y <= 1300):
        GM_skull_sum += GM_lambda[np.where(GM_interp_wv == y)] - skull_lambda[np.where(skull_interp_wv == y)]
        overlap_length += 1

GM_skull_avg_offset = GM_skull_sum/(overlap_length)

# b
GM_avg_offset = (GM_scalp_avg_offset + GM_skull_avg_offset)/2.0

scalp_slope = np.gradient(scalp_lambda)
skull_slope = np.gradient(skull_lambda)

# m
GM_avg_slope = (scalp_slope[np.where(scalp_interp_wv == 1550)] + skull_slope[np.where(skull_interp_wv == 1550)])/2.0

GM_1550 = (GM_avg_slope*1550)+GM_avg_offset

### Plots

fig, ax1 = plt.subplots()
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Absorption Coefficient cm^-1')
ax1.scatter(scalp_wavelengths, scalp_absorption, label='Scalp')
ax1.scatter(skull_wavelengths, skull_absorption, label='Skull Bone')
ax1.scatter(GM_wavelengths, GM_absorption, label='Gray Matter')
ax1.scatter(WM_wavelengths, WM_absorption, label='White Matter')

ax1.plot(scalp_interp_wv, scalp_lambda, '--')
ax1.plot(skull_interp_wv, skull_lambda, '--')
ax1.plot(GM_interp_wv, GM_lambda, '--')
ax1.plot(WM_interp_wv, WM_lambda, '--')
ax1.plot(1550, GM_1550, marker='o')

ax1.legend(loc='upper right')

plt.show()