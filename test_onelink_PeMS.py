import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C, RBF
from sklearn import preprocessing
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

'''单link测试，有滑动窗口,五折交叉验证,pems'''

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
def train_data(dir,datet,id,window,step):
    x_data=[]
    y_data=[]
    print("训练日期：")
    print(datet)
    for m in range(len(datet)):
        data = []
        d=datet[m]
        path=dir+'link%s_%s.csv' % (id,d)
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
    x_data=np.array(x_data)
    x_data =np.hstack((preprocessing.scale(x_data[:,0:2]),x_data[:,2:3]))
    y_data=np.array(y_data)
    return x_data,y_data
def test_data(dir,d,id,window,step):
    data = []
    x_data=[]
    y_data=[]
    print("测试日期：")
    print(d)
    path =dir+'link%s_%s.csv' % (id,d)
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
    return x_data,y_data
def main(dir,id,window,step,mode):
    date = ["07", "08", "09", "10", "11"]
    result = np.array([])
    truevalue = np.array([])
    err=np.array([])
    resultrf=-np.array([])
    mape5 = []
    mae5 = []
    rmse5 = []
    for i in date:
        datet = date[:]
        datet.remove(i)
        train_x, train_y = train_data(dir, datet, id, window, step)
        test_x, test_y = test_data(dir, i, id, window, window)
        if mode == 0:
            rf = RandomForestRegressor(n_estimators=80, random_state=123, n_jobs=8, max_depth=4)
            rf.fit(train_x, train_y)
            resultrf = rf.predict(test_x)
        if mode == 1:
            kernel1 = C(constant_value=0.1, constant_value_bounds=(1e-10, 1e10)) * Matern(length_scale=0.5,
                                                                                          length_scale_bounds=(
                                                                                              1e-10, 1e10),
                                                                                          nu=1.5)+WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))

            gp = GaussianProcessRegressor(kernel=kernel1, normalize_y=True, n_restarts_optimizer=50, random_state=123)
            gp.fit(train_x, train_y)
            resultrf, sigma = gp.predict(test_x, return_std=True)
            err = np.hstack((err, sigma))
        result = np.hstack((result, resultrf))
        truevalue = np.hstack((truevalue, test_y))
        mape1 = getMAPE(test_y, resultrf)
        mae1 = getMAE(test_y, resultrf)
        rmse1 = getRMSE(test_y, resultrf)
        mape5.append(mape1)
        mae5.append(mae1)
        rmse5.append(rmse1)
    # mape = getMAPE(truevalue, result)
    # mae = getMAE(truevalue, result)
    # rmse = getRMSE(truevalue, result)
    mape = np.mean(mape5)
    mae = np.mean(mae5)
    rmse = np.mean(rmse5)
    plt.figure()
    plt.plot(np.arange(len(result)), truevalue, 'g', label='True value')
    if mode == 0:
        plt.plot(np.arange(len(result)), result, 'r', linestyle='--', label='RandomForestRegressor')
        plt.suptitle("The result of RandomForestRegressor")
    if mode == 1:
        plt.fill(np.concatenate([np.arange(len(result)), np.arange(len(result))[::-1]]),
                 np.concatenate([result + 1.96*err,
                                 (result - 1.96*err)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.plot(np.arange(len(result)), result, 'r', label='GaussianProcessRegressor')
        plt.suptitle("The result of Gaussian Process Regressor")
    plt.title('MAPE: %s, MAE: %.2f, RMSE:%.2f' % ('{:.2f}%'.format(mape * 100), mae, rmse))
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dir = r'D:/transfer_finally/PeMS/Station_data/'
    # id = 801230
    id=818609
    window = 12
    step = 12
    mode=1
    main(dir,id,window,step,mode)
