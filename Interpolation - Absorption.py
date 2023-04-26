# Linear and Polynomial regression did not yield promising results
# import statistics
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import numpy as np
import matplotlib as mpl
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


### Known data

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

Water_absorption = [0.00058, 0.00038, 0.00028, 0.000247, 0.00025, 0.00032, 0.00045, 0.00079, 0.0023, 0.0028, 0.0032, 0.00415, 0.006, 0.0159, 0.026, 0.024, 0.02, 0.019858, 0.023907, 0.028, 0.029069, 0.034707, 0.043, 0.046759, 0.051999, 0.056, 0.055978, 0.060432, 0.068, 0.072913, 0.10927, 0.144, 0.17296, 0.26737, 0.39, 0.42, 0.45, 0.45,
                    0.43, 0.41, 0.36, 0.27, 0.16, 0.12, 0.13, 0.17, 0.52, 0.66, 0.89, 1.04, 1.04, 0.95, 0.88, 0.89, 0.98, 1.11, 1.38, 1.83, 2.77, 6.1, 12.39, 22.12, 28.8, 28.4, 21.23, 17.59, 14.05, 11.83, 9.67, 7.95, 6.72, 5.82, 4.98, 4.54, 4.49, 4.44, 5.11, 6.14, 7.14, 8.12, 8.03, 8.98, 10.24, 14.19, 31.08, 66.14, 114.54, 119.83, 105.79, 92.03, 69.12]
Water_wavelengths = [400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 810, 820, 825, 830, 840, 850, 860, 870, 875, 880, 890, 900, 910, 920, 925, 930, 940, 950, 960, 970, 975, 980, 990, 1000, 1020, 1040, 1060, 1080, 1100, 1120,
                     1140, 1160, 1180, 1200, 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380, 1400, 1420, 1440, 1460, 1480, 1500, 1520, 1540, 1560, 1580, 1600, 1620, 1640, 1660, 1680, 1700, 1720, 1740, 1760, 1780, 1800, 1820, 1840, 1860, 1880, 1900, 1920, 1940, 1960, 1980, 2000]

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

### Gray Matter Vertical Offset

GM_scalp_sum = 0
overlap_length = 0

# Avg scalp vertical offset over 805 nm to 1300 nm - Gray Matter
for x in scalp_interp_wv:
    if (x >= 805) and (x <= 1300):
        GM_scalp_sum += GM_lambda[np.where(GM_interp_wv == x)] - scalp_lambda[np.where(scalp_interp_wv == x)]
        overlap_length += 1

GM_scalp_avg_offset = GM_scalp_sum/(overlap_length)

GM_skull_sum = 0
overlap_length = 0

# Avg skull vertical offset over 801 nm to 1300 nm - Gray Matter
for y in skull_interp_wv:
    if (y >= 801) and (y <= 1300):
        GM_skull_sum += GM_lambda[np.where(GM_interp_wv == y)] - skull_lambda[np.where(skull_interp_wv == y)]
        overlap_length += 1

GM_skull_avg_offset = GM_skull_sum/(overlap_length)

# Add avg scalp offset to GM
GM_est_scalp_abs = scalp_absorption + GM_scalp_avg_offset

# Add avg skull offset to GM
GM_est_skull_abs = skull_absorption + GM_skull_avg_offset

### White Matter Vertical Offset

WM_scalp_sum = 0
overlap_length = 0

# Avg scalp vertical offset over 805 nm to 1300 nm - White Matter
for x in scalp_interp_wv:
    if (x >= 805) and (x <= 1300):
        WM_scalp_sum += WM_lambda[np.where(WM_interp_wv == x)] - scalp_lambda[np.where(scalp_interp_wv == x)]
        overlap_length += 1

WM_scalp_avg_offset = WM_scalp_sum/(overlap_length)

WM_skull_sum = 0
overlap_length = 0

# Avg skull vertical offset over 801 nm to 1300 nm - White Matter
for y in skull_interp_wv:
    if (y >= 801) and (y <= 1300):
        WM_skull_sum += WM_lambda[np.where(WM_interp_wv == y)] - skull_lambda[np.where(skull_interp_wv == y)]
        overlap_length += 1

WM_skull_avg_offset = WM_skull_sum/(overlap_length)

# Add avg scalp offset to WM
WM_est_scalp_abs = scalp_absorption + WM_scalp_avg_offset

# Add avg skull offset to WM
WM_est_skull_abs = skull_absorption + WM_skull_avg_offset


### Skull Vertical Offset - Validation using Scalp

skull_scalp_sum = 0
overlap_length = 0

# Avg scalp vertical offset over 801 nm to 1300 nm - Skull
for x in scalp_interp_wv:
    if (x >= 800) and (x <= 1300):
        skull_scalp_sum += skull_lambda[np.where(skull_interp_wv == x)] - scalp_lambda[np.where(scalp_interp_wv == x)]
        overlap_length += 1

skull_scalp_avg_offset = skull_scalp_sum/(overlap_length)

# Add avg scalp offset to Skull
skull_est_scalp_abs = scalp_absorption + skull_scalp_avg_offset



# Interpolate vertically offset data points for scalp and skull - Gray Matter
GM_est_scalp_interp = interp1d(scalp_wavelengths, GM_est_scalp_abs, kind='cubic')
GM_est_skull_interp = interp1d(skull_wavelengths, GM_est_skull_abs, kind='cubic')

# Interpolate vertically offset data points for scalp and skull - White Matter
WM_est_scalp_interp = interp1d(scalp_wavelengths, WM_est_scalp_abs, kind='cubic')
WM_est_skull_interp = interp1d(skull_wavelengths, WM_est_skull_abs, kind='cubic')


# Interpolate vertically offset data points for scalp - Skull
skull_est_scalp_interp = interp1d(scalp_wavelengths, skull_est_scalp_abs, kind='cubic')


# Interpolated functions for estimated Gray Matter scalp and skull - Gray Matter
GM_scalp_estimation = GM_est_scalp_interp(scalp_interp_wv)
GM_skull_estimation = GM_est_skull_interp(skull_interp_wv)

# Interpolated functions for estimated White Matter scalp and skull - White Matter
WM_scalp_estimation = WM_est_scalp_interp(scalp_interp_wv)
WM_skull_estimation = WM_est_skull_interp(skull_interp_wv)


# Interpolated functions for estimated skull scalp - skull
skull_scalp_estimation = skull_est_scalp_interp(scalp_interp_wv)

# Average the interpolated functions - Gray Matter
# Note: Skull wavelength range is limited to match scalp wavelength range (805 nm - 2000 nm)
GM_avg_est_lambda = (GM_scalp_estimation + GM_skull_estimation[4:1999]) / 2

# Average the interpolated functions - White Matter
# Note: Skull wavelength range is limited to match scalp wavelength range (805 nm - 2000 nm)
WM_avg_est_lambda = (WM_scalp_estimation + WM_skull_estimation[4:1999]) / 2

coef_of_determ = r2_score(skull_interpolation(skull_interp_wv)[4:1999], skull_est_scalp_interp(scalp_interp_wv))

print(coef_of_determ)

### Plots - dashed line plots are the interpolated functions

mpl.rc('font', family='Times New Roman')
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels

plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Absorption Coefficient cm^-1', fontsize=12)

### Scalp
plt.plot(scalp_interp_wv, scalp_lambda, ':', label='Scalp', color='#DDCC77')
plt.scatter(scalp_wavelengths, scalp_absorption, color='#DDCC77')

### Skull
plt.plot(skull_interp_wv, skull_lambda, ':', label='Skull Bone', color='#88CCEE')
plt.scatter(skull_wavelengths, skull_absorption, color='#88CCEE')

# ### Gray Matter
# plt.plot(GM_interp_wv, GM_lambda, ':', label='Gray Matter', color='#44AA99')
# plt.scatter(GM_wavelengths, GM_absorption, color='#44AA99')

# ### White Matter
# plt.plot(WM_interp_wv, WM_lambda, ':', label='White Matter', color='#882255')
# plt.scatter(WM_wavelengths, WM_absorption, color='#882255')

# ### Estimation for Gray Matter

# plt.plot(skull_interp_wv, GM_skull_estimation, ls=(5, (10, 3)), label='GM Skull Estimation', color='#117733')
# plt.scatter(skull_wavelengths, GM_est_skull_abs, color='#117733')

# plt.plot(scalp_interp_wv, GM_avg_est_lambda, '-.', label='GM Avg. Scalp and Skull', color='#117733')

# plt.plot(scalp_interp_wv, GM_scalp_estimation, '--', label='GM Scalp Estimation', color='#117733')
# plt.scatter(scalp_wavelengths, GM_est_scalp_abs, color='#117733')

# ### Estimation for White Matter

# plt.plot(skull_interp_wv, WM_skull_estimation, ls=(5, (10, 3)), linewidth=2, label='WM Skull Estimation', color='#332288')
# plt.scatter(skull_wavelengths, WM_est_skull_abs, color='#332288')

# plt.plot(scalp_interp_wv, WM_avg_est_lambda, '-.', linewidth=2, label='WM Avg. Scalp and Skull', color='#332288')

# plt.plot(scalp_interp_wv, WM_scalp_estimation, '--', linewidth=2, label='WM Scalp Estimation', color='#332288')
# plt.scatter(scalp_wavelengths, WM_est_scalp_abs, color='#332288')

### Estimation for Skull using Scalp

plt.plot(skull_interp_wv[4:1999], skull_scalp_estimation, ls=(5, (10, 3)), linewidth=2, label='Skull Scalp Estimation', color='#332288')

plt.axvline(x = 1550, color = '#CC6677', label = '1550 nm')
plt.axvline(x = 1300, color = 'black', label = 'End of GM/WM Known Data', linestyle = '--')

""" Add 1550 nm absorption labels """
# plt.annotate("Scalp Interpolation: " + "{:.3f}".format(scalp_interpolation(1550)) + " cm^-1", xy=(1550, scalp_interpolation(1550)), xytext=(1590, -0.5), arrowprops=dict(arrowstyle="->"))
# plt.annotate("Skull Interpolation: " + "{:.3f}".format(skull_interpolation(1550)) + " cm^-1", xy=(1550, skull_interpolation(1550)), xytext=(1650, 0), arrowprops=dict(arrowstyle="->"))
# plt.annotate("Avg. White Matter Extrapolation: " + "{:.3f}".format(float(WM_avg_est_lambda[np.where(scalp_interp_wv==1550)])) + " cm^-1", xy=(1550, WM_avg_est_lambda[np.where(scalp_interp_wv==1550)]), xytext=(1710, 0.5), arrowprops=dict(arrowstyle="->"))
# plt.annotate("Gray Matter Skull Extrapolation: " + "{:.3f}".format(float(GM_skull_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1", xy=(1550, GM_skull_estimation[np.where(skull_interp_wv==1550)]), xytext=(1780, 1.0), arrowprops=dict(arrowstyle="->"))
# plt.annotate("White Matter Skull Extrapolation: " + "{:.3f}".format(float(WM_skull_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1", xy=(1550, WM_skull_estimation[np.where(skull_interp_wv==1550)]), xytext=(1830, 1.5), arrowprops=dict(arrowstyle="->"))
# plt.annotate("White Matter Scalp Extrapolation: " + "{:.3f}".format(float(WM_scalp_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1", xy=(1550, WM_scalp_estimation[np.where(skull_interp_wv==1550)]), xytext=(1850, 1.5), arrowprops=dict(arrowstyle="->"))
# plt.annotate("Gray Matter Scalp Extrapolation: " + "{:.3f}".format(float(GM_scalp_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1", xy=(1550, GM_scalp_estimation[np.where(skull_interp_wv==1550)]), xytext=(1870, 1.0), arrowprops=dict(arrowstyle="->"))

""" Print out 1550 nm coefficients """
# print("1550")
# print("Scalp Interpolation: " + "{:.3f}".format(scalp_interpolation(1550)) + " cm^-1")
# print("Skull Interpolation: " + "{:.3f}".format(skull_interpolation(1550)) + " cm^-1")
# print("Gray Matter Skull Extrapolation: " + "{:.3f}".format(float(GM_skull_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1")
# print("Avg. Gray Matter Extrapolation: " + "{:.3f}".format(float(GM_avg_est_lambda[np.where(scalp_interp_wv==1550)])) + " cm^-1")
# print("Gray Matter Scalp Extrapolation: " + "{:.3f}".format(float(GM_scalp_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1")
# print("White Matter Skull Extrapolation: " + "{:.3f}".format(float(WM_skull_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1")
# print("Avg. White Matter Extrapolation: " + "{:.3f}".format(float(WM_avg_est_lambda[np.where(scalp_interp_wv==1550)])) + " cm^-1")
# print("White Matter Scalp Extrapolation: " + "{:.3f}".format(float(WM_scalp_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1")

# # Print out 1064 nm coefficients
# print("1064")
# print("Scalp Interpolation: " + "{:.3f}".format(scalp_interpolation(1064)) + " cm^-1")
# print("Skull Interpolation: " + "{:.3f}".format(skull_interpolation(1064)) + " cm^-1")
# print("Gray Matter Skull Extrapolation: " + "{:.3f}".format(float(GM_skull_estimation[np.where(skull_interp_wv==1064)])) + " cm^-1")
# print("Avg. Gray Matter Extrapolation: " + "{:.3f}".format(float(GM_avg_est_lambda[np.where(scalp_interp_wv==1064)])) + " cm^-1")
# print("Gray Matter Scalp Extrapolation: " + "{:.3f}".format(float(GM_scalp_estimation[np.where(skull_interp_wv==1064)])) + " cm^-1")
# print("White Matter Skull Extrapolation: " + "{:.3f}".format(float(WM_skull_estimation[np.where(skull_interp_wv==1064)])) + " cm^-1")
# print("Avg. White Matter Extrapolation: " + "{:.3f}".format(float(WM_avg_est_lambda[np.where(scalp_interp_wv==1064)])) + " cm^-1")
# print("White Matter Scalp Extrapolation: " + "{:.3f}".format(float(WM_scalp_estimation[np.where(skull_interp_wv==1064)])) + " cm^-1")

# # Print out 980 nm coefficients
# print("980")
# print("Scalp Interpolation: " + "{:.3f}".format(scalp_interpolation(980)) + " cm^-1")
# print("Skull Interpolation: " + "{:.3f}".format(skull_interpolation(980)) + " cm^-1")
# print("Gray Matter Skull Extrapolation: " + "{:.3f}".format(float(GM_skull_estimation[np.where(skull_interp_wv==980)])) + " cm^-1")
# print("Avg. Gray Matter Extrapolation: " + "{:.3f}".format(float(GM_avg_est_lambda[np.where(scalp_interp_wv==980)])) + " cm^-1")
# print("Gray Matter Scalp Extrapolation: " + "{:.3f}".format(float(GM_scalp_estimation[np.where(skull_interp_wv==980)])) + " cm^-1")
# print("White Matter Skull Extrapolation: " + "{:.3f}".format(float(WM_skull_estimation[np.where(skull_interp_wv==980)])) + " cm^-1")
# print("Avg. White Matter Extrapolation: " + "{:.3f}".format(float(WM_avg_est_lambda[np.where(scalp_interp_wv==980)])) + " cm^-1")
# print("White Matter Scalp Extrapolation: " + "{:.3f}".format(float(WM_scalp_estimation[np.where(skull_interp_wv==980)])) + " cm^-1")

# # Print out 810 nm coefficients
# print("810")
# print("Scalp Interpolation: " + "{:.3f}".format(scalp_interpolation(810)) + " cm^-1")
# print("Skull Interpolation: " + "{:.3f}".format(skull_interpolation(810)) + " cm^-1")
# print("Gray Matter Skull Extrapolation: " + "{:.3f}".format(float(GM_skull_estimation[np.where(skull_interp_wv==810)])) + " cm^-1")
# print("Avg. Gray Matter Extrapolation: " + "{:.3f}".format(float(GM_avg_est_lambda[np.where(scalp_interp_wv==810)])) + " cm^-1")
# print("Gray Matter Scalp Extrapolation: " + "{:.3f}".format(float(GM_scalp_estimation[np.where(skull_interp_wv==810)])) + " cm^-1")
# print("White Matter Skull Extrapolation: " + "{:.3f}".format(float(WM_skull_estimation[np.where(skull_interp_wv==810)])) + " cm^-1")
# print("Avg. White Matter Extrapolation: " + "{:.3f}".format(float(WM_avg_est_lambda[np.where(scalp_interp_wv==810)])) + " cm^-1")
# print("White Matter Scalp Extrapolation: " + "{:.3f}".format(float(WM_scalp_estimation[np.where(skull_interp_wv==810)])) + " cm^-1")

plt.title('Absorption Coefficient Approximation', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.show()