# Linear and Polynomial regression did not yield promising results
# import statisticsscalp_scattering
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib as mpl
from scipy.interpolate import interp1d
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


### Known data

# Scalp wavelengths (805 - 2000 nm) and scalp scattering

scalp_scattering = [14.09, 15.66, 16.44, 16.83, 17.10, 16.70,
                    14.70, 14.28, 13.15, 14.40, 14.16, 14.71, 13.36, 12.15, 11.52, 12.00]
scalp_wavelengths = [805, 900, 950, 1000, 1100, 1200, 1300,
                     1400, 1430, 1500, 1600, 1700, 1800, 1900, 1930, 2000]

# Skull wavelengths (801 - 2000 nm) and skull scattering

skull_scattering = [19.48, 18.03, 17.38, 17.10, 15.92, 16.53, 16.77,
                         14.78, 17.22, 16.84, 15.96, 15.84, 16.12, 15.82, 15.42, 11.37, 10.92, 11.48]
skull_wavelengths = [801, 900, 980, 1000, 1100, 1180, 1200,
                          1300, 1400, 1465, 1500, 1600, 1700, 1740, 1800, 1900, 1930, 2000]

# Gray matter wavelengths (400 - 1300 nm) and gray matter scattering

GM_scattering = [25.878, 26.593, 26.709, 19.389, 15.957, 15.283, 13.315, 11.367, 10.370, 9.480, 8.907,
                 8.481, 7.886, 7.707, 7.555, 7.351, 7.055, 6.868, 6.059, 5.333, 5.197, 5.070, 4.882, 4.669, 4.560]
GM_wavelengths = [400, 418, 428, 450, 488, 500, 550, 600, 632, 670, 700, 750,
                  800, 830, 850, 870, 900, 950, 1000, 1064, 1100, 1150, 1200, 1250, 1300]

# White matter wavelengths (400 - 1300 nm) and white matter scattering

WM_scattering = [88.611, 83.304, 80.905, 77.053, 70.112, 68.318, 62.383, 56.759, 53.179, 50.067, 47.626,
                 45.061, 41.878, 40.634, 39.658, 38.785, 37.607, 35.851, 32.603, 30.161, 29.219, 27.951, 26.646, 25.310, 24.250]
WM_wavelengths = [400, 418, 428, 450, 488, 500, 550, 600, 632, 670, 700, 750,
                  800, 830, 850, 870, 900, 950, 1000, 1064, 1100, 1150, 1200, 1250, 1300]

### Interpolation

# Interpolate existing data
scalp_interpolation = interp1d(scalp_wavelengths, scalp_scattering, kind='cubic')
skull_interpolation = interp1d(skull_wavelengths, skull_scattering, kind='cubic')
GM_interpolation = interp1d(GM_wavelengths, GM_scattering, kind='cubic')
WM_interpolation = interp1d(WM_wavelengths, WM_scattering, kind='cubic')

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
GM_est_scalp_abs = scalp_scattering + GM_scalp_avg_offset

# Add avg skull offset to GM
GM_est_skull_abs = skull_scattering + GM_skull_avg_offset

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
WM_est_scalp_abs = scalp_scattering + WM_scalp_avg_offset

# Add avg skull offset to WM
WM_est_skull_abs = skull_scattering + WM_skull_avg_offset

# Interpolate vertically offset data points for scalp and skull - Gray Matter
GM_est_scalp_interp = interp1d(scalp_wavelengths, GM_est_scalp_abs, kind='cubic')
GM_est_skull_interp = interp1d(skull_wavelengths, GM_est_skull_abs, kind='cubic')

# Interpolate vertically offset data points for scalp and skull - White Matter
WM_est_scalp_interp = interp1d(scalp_wavelengths, WM_est_scalp_abs, kind='cubic')
WM_est_skull_interp = interp1d(skull_wavelengths, WM_est_skull_abs, kind='cubic')

# Interpolated functions for estimated Gray Matter scalp and skull - Gray Matter
GM_scalp_estimation = GM_est_scalp_interp(scalp_interp_wv)
GM_skull_estimation = GM_est_skull_interp(skull_interp_wv)

# Interpolated functions for estimated Gray Matter scalp and skull - White Matter
# 5.52 constant is used to align the predicted WM data with the last point of the 
# known WM data. This provides a better fit of the data.
WM_scalp_estimation = WM_est_scalp_interp(scalp_interp_wv) - 5.52
WM_skull_estimation = WM_est_skull_interp(skull_interp_wv) - 5.52


# Average the interpolated functions - Gray Matter
# Note: Skull wavelength range is limited to match scalp wavelength range (805 nm - 2000 nm)
GM_avg_est_lambda = (GM_scalp_estimation + GM_skull_estimation[4:1999]) / 2

# Average the interpolated functions - White Matter
# Note: Skull wavelength range is limited to match scalp wavelength range (805 nm - 2000 nm)
WM_avg_est_lambda = (WM_scalp_estimation + WM_skull_estimation[4:1999]) / 2

### Plots - dashed line plots are the interpolated functions

mpl.rc('font', family='Times New Roman')

plt.xlabel('Wavelength (nm)', fontsize='xx-large')
plt.ylabel('scattering Coefficient cm^-1', fontsize='xx-large')

### Scalp
plt.plot(scalp_interp_wv, scalp_lambda, ':', label='Scalp', color='#DDCC77')
plt.scatter(scalp_wavelengths, scalp_scattering, color='#DDCC77')

### Skull
plt.plot(skull_interp_wv, skull_lambda, ':', label='Skull Bone', color='#88CCEE')
plt.scatter(skull_wavelengths, skull_scattering, color='#88CCEE')

### Gray Matter
plt.plot(GM_interp_wv, GM_lambda, ':', label='Gray Matter', color='#44AA99')
plt.scatter(GM_wavelengths, GM_scattering, color='#44AA99')

### White Matter
plt.plot(WM_interp_wv, WM_lambda, ':', label='White Matter', color='#882255')
plt.scatter(WM_wavelengths, WM_scattering, color='#882255')

### Estimation for Gray Matter

plt.plot(skull_interp_wv, GM_skull_estimation, ls=(5, (10, 3)), label='GM Skull Estimation', color='#117733')
plt.scatter(skull_wavelengths, GM_est_skull_abs, color='#117733')

plt.plot(scalp_interp_wv, GM_avg_est_lambda, '-.', label='GM Avg b/t Scalp and Skull', color='#117733')

plt.plot(scalp_interp_wv, GM_scalp_estimation, '--', label='GM Scalp Estimation', color='#117733')
plt.scatter(scalp_wavelengths, GM_est_scalp_abs, color='#117733')

### Estimation for White Matter

plt.plot(skull_interp_wv, WM_skull_estimation, ls=(5, (10, 3)), linewidth=2, label='WM Skull Estimation', color='#332288')
plt.scatter(skull_wavelengths, WM_est_skull_abs - 5.52, color='#332288')

plt.plot(scalp_interp_wv, WM_avg_est_lambda, '-.', linewidth=2, label='WM Avg b/t Scalp and Skull', color='#332288')

plt.plot(scalp_interp_wv, WM_scalp_estimation, '--', linewidth=2, label='WM Scalp Estimation', color='#332288')
plt.scatter(scalp_wavelengths, WM_est_scalp_abs - 5.52, color='#332288')

plt.axvline(x = 1550, color = '#CC6677', label = '1550 nm')
plt.axvline(x = 1300, color = 'black', label = 'End of GM/WM Known Data', linestyle = '--')

# Add 1550 nm scattering labels
# plt.annotate("Scalp Interpolation: " + "{:.3f}".format(scalp_interpolation(1550)) + " cm^-1", xy=(1550, scalp_interpolation(1550)), xytext=(1590, -0.5), arrowprops=dict(arrowstyle="->"))
# plt.annotate("Skull Interpolation: " + "{:.3f}".format(skull_interpolation(1550)) + " cm^-1", xy=(1550, skull_interpolation(1550)), xytext=(1650, 0), arrowprops=dict(arrowstyle="->"))
# plt.annotate("Avg. White Matter Extrapolation: " + "{:.3f}".format(float(WM_avg_est_lambda[np.where(scalp_interp_wv==1550)])) + " cm^-1", xy=(1550, WM_avg_est_lambda[np.where(scalp_interp_wv==1550)]), xytext=(1710, 0.5), arrowprops=dict(arrowstyle="->"))
# plt.annotate("Gray Matter Skull Extrapolation: " + "{:.3f}".format(float(GM_skull_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1", xy=(1550, GM_skull_estimation[np.where(skull_interp_wv==1550)]), xytext=(1780, 1.0), arrowprops=dict(arrowstyle="->"))
# plt.annotate("White Matter Skull Extrapolation: " + "{:.3f}".format(float(WM_skull_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1", xy=(1550, WM_skull_estimation[np.where(skull_interp_wv==1550)]), xytext=(1830, 1.5), arrowprops=dict(arrowstyle="->"))

print("Scalp Interpolation: " + "{:.3f}".format(scalp_interpolation(1550)) + " cm^-1")
print("Skull Interpolation: " + "{:.3f}".format(skull_interpolation(1550)) + " cm^-1")
print("Gray Matter Skull Extrapolation: " + "{:.3f}".format(float(GM_skull_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1")
print("Avg. Gray Matter Extrapolation: " + "{:.3f}".format(float(GM_avg_est_lambda[np.where(scalp_interp_wv==1550)])) + " cm^-1")
print("Gray Matter Scalp Extrapolation: " + "{:.3f}".format(float(GM_scalp_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1")
print("White Matter Skull Extrapolation: " + "{:.3f}".format(float(WM_skull_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1")
print("Avg. White Matter Extrapolation: " + "{:.3f}".format(float(WM_avg_est_lambda[np.where(scalp_interp_wv==1550)])) + " cm^-1")
print("White Matter Scalp Extrapolation: " + "{:.3f}".format(float(WM_scalp_estimation[np.where(skull_interp_wv==1550)])) + " cm^-1")

# plt.title('Scattering Coefficient Approximation', fontsize='xx-large')
# plt.legend(loc='best', fontsize='xx-large')
# plt.show()