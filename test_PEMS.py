import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel,ConstantKernel as C
from sklearn import metrics
def getRMSE(y_label, pre):
    squaredError = []
    for i in range(len(y_label)):
        squaredError.append((y_label[i] - pre[i]) * (y_label[i] - pre[i]))
    rmse = np.sqrt(sum(squaredError) / len(squaredError))
    return rmse
def getMAPE(y_label, pre):
    return np.mean(np.abs((y_label - pre) / y_label))
def getMAE(y_label, pre):
    return metrics.mean_absolute_error(y_label, pre)
data_read=np.load("PEMS08//pems08.npz")
data=data_read['data']
print(data.shape)
print(data)
oneday=data[0:288,0,:]
train_x=data[0:1440,0,1:]
test_x=data[1440:1728,0,1:]
train_y=data[0:1440,0,0]
test_y=data[1440:1728,0,0]
# print(oneday.shape)
# flow=oneday[:,0]
# speed=oneday[:,1]
# occupancy=oneday[:,2]
# print(flow)
# x=[]
# for i in range(5,1441,5):
#     x.append(i)
# print(len(x))
# plt.figure()
# plt.plot(x, flow, 'blue', linestyle='-')
# plt.show()
# plt.figure()
# plt.plot(x, speed, 'g', linestyle='-')
# plt.show()
# plt.figure()
# plt.plot(x, occupancy, 'r', linestyle='-')
# plt.show()
# kernel1 = C(constant_value=0.1, constant_value_bounds=(1e-10, 1e10)) * Matern(length_scale=0.5,
#                                                                                           length_scale_bounds=(
#                                                                                               1e-10, 1e10),
#                                                                                           nu=1.5)+WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))
# gp = GaussianProcessRegressor(kernel=kernel1, normalize_y=True, n_restarts_optimizer=25, random_state=123)
# gp.fit(train_x, train_y)
# result, sigma = gp.predict(test_x, return_std=True)
# mape = getMAPE(test_y, result)
# mae = getMAE(test_y, result)
# rmse = getRMSE(test_y, result)
# print(mape)
# print(mae)
# print(rmse)
# plt.figure()
# # plt.fill(np.concatenate([np.arange(len(result)), np.arange(len(result))[::-1]]),
# #                  np.concatenate([result + 1.96*sigma,
# #                                  (result - 1.96*sigma)[::-1]]),
# #                  alpha=.5, fc='b', ec='None', label='95% confidence interval')
# plt.plot(np.arange(len(result)), test_y, 'g', label='True value')
# plt.plot(np.arange(len(result)), result, 'r', label='GaussianProcessRegressor')
# plt.fill(np.concatenate([np.arange(len(result)), np.arange(len(result))[::-1]]),
#              np.concatenate([result + 1.96 * sigma,
#                              (result - 1.96 * sigma)[::-1]]),
#              alpha=.5, fc='Silver', ec='None', label='95% confidence interval')
# plt.legend()
# plt.show()