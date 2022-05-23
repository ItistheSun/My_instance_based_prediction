import numpy as np
import csv
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn import preprocessing
from sklearn import metrics
from sklearn.utils.optimize import _check_optimize_result
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C, RBF
import warnings
warnings.filterwarnings("ignore")

'''迁移学习-多link测试，有滑动窗口,五折交叉验证，单link预处理'''
def getRMSE(y_label, pre):
    squaredError = []
    for i in range(len(y_label)):
        squaredError.append((y_label[i] - pre[i]) * (y_label[i] - pre[i]))
    rmse = np.sqrt(sum(squaredError) / len(squaredError))
    return rmse
class MyGPR(GaussianProcessRegressor):
    def __init__(self, *args, max_iter=1000, gtol=1e-06, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_iter = max_iter
        self._gtol = gtol
    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(
                obj_func, initial_theta, method="L-BFGS-B", jac=True,
                bounds=bounds, options={'maxiter':self._max_iter, 'gtol': self._gtol})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = \
                self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)

        return theta_opt, func_min

def getMAPE(y_label, pre):
    return np.mean(np.abs((y_label - pre) / y_label))


def getMAE(y_label, pre):
    return metrics.mean_absolute_error(y_label, pre)


def train_data(dir, datet, similarlink, window, step):
    x_data = []
    y_data = []
    for i in range(len(similarlink)):
        data1 = []
        id = similarlink[i]
        for m in range(len(datet)):
            data = []
            d = datet[m]
            path = dir + 'Station_data/link%s_%s.csv' % (id, d)
            with open(path) as f:
                f_csv = csv.reader(f)
                for row in f_csv:
                    for i in range(len(row)):
                        row[i] = float(row[i])
                    row.append(row[0] // 10)  # 时间片
                    data.append(row[1:])
            m = 0
            n = m + window
            while n <= len(data):
                temp = data[m:n]
                temp = np.array(temp)
                temp = np.sum(temp, axis=0)
                data1.append(temp[1:4])
                y_data.append(temp[0])
                m = m + step
                n = m + window
        data1 = np.array(data1)
        data1 = np.hstack((preprocessing.scale(data1[:, 0:2]), data1[:, 2:3]))
        x_data.append(data1)
    for i in range(len(x_data)):
        if i == 0:
            data2 = x_data[i]
        else:
            data2 = np.vstack((data2, x_data[i]))
    x_data = np.array(data2)
    y_data = np.array(y_data)
    return x_data, y_data


def test_data(dir, datet, id, window, step):  # 待测试link所有天的数据
    x_data = []
    y_data = []
    for m in range(len(datet)):
        data = []
        d = datet[m]
        path = dir + 'Station_data/link%s_%s.csv' % (id, d)
        with open(path) as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                for i in range(len(row)):
                    row[i] = float(row[i])
                row.append(row[0] // 10)  # 时间片
                data.append(row[1:])
        m = 0
        n = m + window
        while n <= len(data):
            temp = data[m:n]
            temp = np.array(temp)
            temp = np.sum(temp, axis=0)
            x_data.append(temp[1:4])
            y_data.append(temp[0])
            m = m + step
            n = m + window
    x_data = np.array(x_data)
    x_data = np.hstack((preprocessing.scale(x_data[:, 0:2]), x_data[:, 2:3]))
    y_data = np.array(y_data)
    return x_data, y_data


def network_test(dir, id, similarlink, window, step, date):
    train_x, train_y = train_data(dir, date, similarlink, window, step)
    test_x, test_y = test_data(dir, date, id, window, window)
    kernel1 = C(constant_value=0.1, constant_value_bounds=(1e-10, 1e10)) * Matern(length_scale=0.5,
                                                                                  length_scale_bounds=(
                                                                                      1e-10, 1e10),
                                                                                  nu=1.5) + WhiteKernel(
        noise_level=1.0, noise_level_bounds=(1e-10, 1e10))

    gp = GaussianProcessRegressor(kernel=kernel1, normalize_y=True, n_restarts_optimizer=50, random_state=123)
    # gp=MyGPR(kernel=kernel1, normalize_y=True, n_restarts_optimizer=50, random_state=123)
    gp.fit(train_x, train_y)
    resultrf, sigma = gp.predict(test_x, return_std=True)
    # rf = RandomForestRegressor(n_estimators=500, random_state=123, n_jobs=8, max_depth=10)
    # rf.fit(train_x, train_y)
    # resultrf = rf.predict(test_x)
    mape = getMAPE(test_y, resultrf)
    mae = getMAE(test_y, resultrf)
    rmse = getRMSE(test_y, resultrf)
    return mape, mae, rmse, resultrf, test_y, sigma

def main():
    date = ["07", "08", "09", "10", "11"]
    dir = r'D:/transfer_finally/PeMS/'
    linktest = [825899]
    s = [[813316, 819684, 801255, 824105, 807818]]
    # linktest = [801595]
    # s = [[824044, 818157, 818549, 825334]]
    window = 12
    step = 12
    mapeall = []
    maeall = []
    rmseall = []
    plt.figure()
    index = 0
    for i in range(len(linktest)):
        index = index + 1
        id = linktest[i]
        similarlink = s[i]
        mape, mae, rmse, resultrf, test_y, err = network_test(dir, id, similarlink, window, step, date)
        mapeall.append(mape)
        maeall.append(mae)
        rmseall.append(rmse)
        tx = plt.subplot(1, 1, index)
        plt.sca(tx)
        plt.plot(np.arange(len(resultrf)), test_y, 'g', label='true value')
        plt.fill(np.concatenate([np.arange(len(resultrf)), np.arange(len(resultrf))[::-1]]),
                 np.concatenate([resultrf + 1.96 * err,
                                 (resultrf - 1.96 * err)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.plot(np.arange(len(resultrf)), resultrf, 'r', linestyle='--', label='result')
        plt.title('Link %s MAPE: %s' % (id, '{:.2f}%'.format(mape * 100)))
        plt.legend()
        print('link %s :mape: %s mae: %f rmse:%f' % (id, '{:.2f}%'.format(mape * 100), mae, rmse))
    plt.suptitle("The result of the instance-based transfer learning",fontdict={'fontsize': 14})
    print("平均mape：%s" % (np.mean(mapeall)))
    print("平均mae：%s" % (np.mean(maeall)))
    print("平均rmse：%s" % (np.mean(rmseall)))
    plt.show()



if __name__ == '__main__':
    main()

