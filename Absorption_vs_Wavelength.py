import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statistics

scalp_absorption = [0.52, 0.40, 0.39, 0.33, 0.19, 0.65,
                    0.50, 1.98, 2.19, 2.04, 1.43, 1.87, 1.73, 2.57, 2.52, 2.09]
scalp_wavelengths = [805, 900, 950, 1000, 1100, 1200, 1300,
                     1400, 1430, 1500, 1600, 1700, 1800, 1900, 1930, 2000]

skull_bone_absorption = [0.11, 0.15, 0.23, 0.22, 0.16, 0.67, 0.67,
                         0.54, 2.43, 3.33, 3.13, 2.47, 2.77, 2.98, 2.97, 4.39, 4.97, 4.47]
skull_bone_wavelengths = [801, 900, 980, 1000, 1100, 1180, 1200,
                          1300, 1400, 1465, 1500, 1600, 1700, 1740, 1800, 1900, 1930, 2000]

# GM_absorption = [9.778, 14.873, 16.722, 5.161, 2.272, 2.206, 2.955, 1.460, 0.925, 0.809, 0.733,
#                  0.599, 0.507, 0.485, 0.472, 0.479, 0.503, 0.521, 0.585, 0.502, 0.502, 0.815, 1.010, 0.865, 0.894]
# GM_wavelengths = [400, 418, 428, 450, 488, 500, 550, 600, 632, 670, 700, 750,
#                   800, 830, 850, 870, 900, 950, 1000, 1064, 1100, 1150, 1200, 1250, 1300]

# WM_absorption = [9.134, 13.603, 15.417, 3.958, 1.869, 1.834, 2.584, 1.175, 0.801, 0.711, 0.674,
#                  0.649, 0.622, 0.626, 0.643, 0.666, 0.684, 0.785, 0.883, 0.752, 0.762, 1.135, 1.420, 1.268, 1.274]
# WM_wavelengths = [400, 418, 428, 450, 488, 500, 550, 600, 632, 670, 700, 750,
#                   800, 830, 850, 870, 900, 950, 1000, 1064, 1100, 1150, 1200, 1250, 1300]

GM_800_abs = [0.507, 0.485, 0.472, 0.479, 0.503, 0.521,
              0.585, 0.502, 0.502, 0.815, 1.010, 0.865, 0.894]
GM_800_wv = [800, 830, 850, 870, 900, 950,
             1000, 1064, 1100, 1150, 1200, 1250, 1300]

WM_800_abs = [0.622, 0.626, 0.643, 0.666, 0.684, 0.785,
              0.883, 0.752, 0.762, 1.135, 1.420, 1.268, 1.274]
WM_800_wv = [800, 830, 850, 870, 900, 950,
             1000, 1064, 1100, 1150, 1200, 1250, 1300]

# Multiple linear regression

# X array - known coefficients

scalp_linreg = np.array([0.52, 0.40, 0.39, 0.33, 0.19, 0.65, 0.50, 1.98, 2.19, 2.04, 1.43, 1.87, 1.73])
# scalp_wavelengths = [805, 900, 950, 1000, 1100, 1200, 1300,
#                      1400, 1430, 1500, 1600, 1700, 1800]

skull_linreg = np.array([0.11, 0.15, 0.23, 0.22, 0.16, 0.67, 0.54, 2.43, 3.33, 3.13, 2.47, 2.77, 2.97])
# skull_bone_wavelengths = [801, 900, 980, 1000, 1100, 1200,
#                           1300, 1400, 1465, 1500, 1600, 1700, 1740, 1800]

# Two dimensional X array
twoD_scalp_skull = np.vstack((scalp_linreg,skull_linreg)).T

# Transform X array to polynomial fit

transformer = PolynomialFeatures(degree=2, include_bias=False)

transformer.fit(twoD_scalp_skull)

X_ = transformer.transform(twoD_scalp_skull)
# print(X_)

# Y array - unknown coefficients

brain_absorption = [statistics.mean(k) for k in zip(GM_800_abs, WM_800_abs)]
brain_wavelengths = [800, 830, 850, 870, 900, 950,
            1000, 1064, 1100, 1150, 1200, 1250, 1300]

model = LinearRegression().fit(twoD_scalp_skull, brain_absorption)
model1 = LinearRegression().fit(X_,brain_absorption)

# r_sq = model.score(twoD_scalp_skull,brain_absorption) # Linear Regression
# Linear regression r_sq = 0.704422
r_sq = model1.score(X_,brain_absorption)
print('coefficient of determintation: %f' %(r_sq))
# Polynomial regression r_sq = 0.893192

abs_predict2 = model.predict(twoD_scalp_skull) # Linear Regression
abs_predict1 = model1.predict(X_) # Polynomial Regression
predict_wavelengths = [801, 900, 980, 1000, 1100, 1200,
                          1300, 1400, 1465, 1500, 1600, 1700, 1740]

fig, ax1 = plt.subplots()
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Absorption Coefficient cm^-1')
ax1.plot(scalp_wavelengths, scalp_absorption, label='Scalp')
ax1.plot(skull_bone_wavelengths, skull_bone_absorption, label='Skull Bone')
ax1.plot(brain_wavelengths, brain_absorption, label='Brain')
ax1.plot(predict_wavelengths, abs_predict2, label='LinReg')
ax1.plot(predict_wavelengths, abs_predict1, label='PolyReg')
ax1.legend(loc='upper left')

plt.show()