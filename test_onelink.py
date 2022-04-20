import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from sklearn import preprocessing
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

'''单link测试，有滑动窗口,五折交叉验证'''
lane = {"link171": 3, "link60": 2, "link7": 1, "link26": 3, "link8": 3, "link4": 3, "link17": 3, "link162": 3,
        "link14": 1, "link56": 3, "link19": 3, "link81": 3, "link72": 2, "link15": 2, "link5": 2, "link65": 3,
        "link12": 2, "link75": 2, "link156": 2, "link153": 2, "link20": 1, "link47": 3, "link11": 1, "link10": 3,
        "link18": 3, "link28": 3, "link170": 3, "link154": 2, "link6": 2, "link21": 2, "link161": 3, "link157": 2,
        "link13": 1, "link116": 3, "link3": 2, "link9": 3, "link27": 3, "link16": 1, "link1": 3, "link2": 2}
linkID = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 26, 27, 28, 47, 56, 65, 72, 75,
          81, 116, 153, 154, 156, 157, 161, 162, 170, 171}
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
            x_data.append(temp[1:5])
            y_data.append(temp[0])
            m = m + step
            n = m + window
    x_data=np.array(x_data)
    # x_data = x_data.reshape(-1, 1)
    # x_data=preprocessing.scale(x_data)

    x_data =np.hstack((preprocessing.scale(x_data[:,0:3]),x_data[:,3:4]))
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
        x_data.append(temp[1:5])
        y_data.append(temp[0])
        m = m + step
        n = m + window
    x_data = np.array(x_data)
    x_data = np.hstack((preprocessing.scale(x_data[:, 0:3]), x_data[:, 3:4]))
    y_data = np.array(y_data)
    return x_data,y_data
def main(dir,id,window,step,mode):
    date = ["082", "083", "084", "085", "086"]
    result = np.array([])
    truevalue = np.array([])
    err=np.array([])
    resultrf=-np.array([])
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
            # n_features=4
            # kernel3=C(constant_value=0.1, constant_value_bounds=(1e-10, 1e10)) *Matern(
            #     np.ones(n_features)*1, tuple([(0.1, 5)] * n_features),
            #     nu=2.5) + WhiteKernel(0.5, 'fixed')
            # print(kernel3)
            gp = GaussianProcessRegressor(kernel=kernel1, normalize_y=True, n_restarts_optimizer=25, random_state=123)
            gp.fit(train_x, train_y)
            resultrf, sigma = gp.predict(test_x, return_std=True)
            err = np.hstack((err, sigma))
        result = np.hstack((result, resultrf))
        truevalue = np.hstack((truevalue, test_y))
    mape = getMAPE(truevalue, result)
    mae = getMAE(truevalue, result)
    rmse = getRMSE(truevalue, result)
    plt.figure()
    plt.plot(np.arange(len(result)), truevalue, 'g', label='True value')
    if mode == 0:
        plt.plot(np.arange(len(result)), result, 'r', linestyle='--', label='RandomForestRegressor')
        plt.suptitle("The result of RandomForestRegressor")
    if mode == 1:
        print(err)
        # plt.fill_between(np.arange(len(result)), result + 1.96*err, result - 1.96*err, facecolor='blue',alpha=0.5,label='95% confidence interval')
        plt.fill(np.concatenate([np.arange(len(result)), np.arange(len(result))[::-1]]),
                 np.concatenate([result + 1.96*err,
                                 (result - 1.96*err)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% confidence interval')
        plt.plot(np.arange(len(result)), result, 'r', label='GaussianProcessRegressor')
        plt.suptitle("The result of Gaussian Process Regressor")
    plt.title('MAPE: %s, MAE: %.2f, RMSE:%.2f' % ('{:.2f}%'.format(mape * 100), mae, rmse))
    plt.legend()
    plt.show()
def com_figure(dir,id,window,step):
    date = ["082", "083", "084", "085", "086"]
    result1 = np.array([])
    result2 = np.array([])
    result3 = np.array([])
    result4 = np.array([])
    truevalue = np.array([])
    err=np.array([])
    resultrf=np.array([])
    for i in date:
        datet = date[:]
        datet.remove(i)
        train_x, train_y = train_data(dir, datet, id, window, step)
        test_x, test_y = test_data(dir, i, id, window, window)
        regr = GridSearchCV(svm.SVR(kernel='rbf',gamma=0.1),cv=5,param_grid={"C":[1e0,1e1,1e2,1e3],
                                                                             "gamma":np.logspace(-2,2,5)})
        regr.fit(train_x, train_y)
        resultsvr=regr.predict(test_x)
        rf = RandomForestRegressor(n_estimators=80, random_state=123, n_jobs=8, max_depth=4)
        rf.fit(train_x, train_y)
        resultrf = rf.predict(test_x)
        knn=KNeighborsRegressor(n_neighbors=2)
        knn.fit(train_x, train_y)
        resultknn=knn.predict(test_x)
        kernel1 = C(constant_value=0.1, constant_value_bounds=(1e-10, 1e10)) * Matern(length_scale=0.5,
                                                                                      length_scale_bounds=(
                                                                                          1e-10, 1e10),
                                                                                      nu=1.5) + WhiteKernel(
            noise_level=1.0, noise_level_bounds=(1e-5, 1e5))
        gp = GaussianProcessRegressor(kernel=kernel1, normalize_y=True, n_restarts_optimizer=25, random_state=123)
        gp.fit(train_x, train_y)
        resultgp, sigma = gp.predict(test_x, return_std=True)
        err = np.hstack((err, sigma))

        result1 = np.hstack((result1, resultsvr))
        result2 = np.hstack((result2, resultrf))
        result3 = np.hstack((result3, resultknn))
        result4 = np.hstack((result4, resultgp))
        truevalue = np.hstack((truevalue, test_y))
    mape1 = getMAPE(truevalue, result1)
    mae1 = getMAE(truevalue, result1)
    rmse1 = getRMSE(truevalue, result1)
    mape2 = getMAPE(truevalue, result2)
    mae2 = getMAE(truevalue, result2)
    rmse2 = getRMSE(truevalue, result2)
    mape3 = getMAPE(truevalue, result3)
    mae3 = getMAE(truevalue, result3)
    rmse3 = getRMSE(truevalue, result3)
    mape4= getMAPE(truevalue, result4)
    mae4= getMAE(truevalue, result4)
    rmse4 = getRMSE(truevalue, result4)
    print("SVR:", end="")
    print(mape1, mae1, rmse1)
    print("RF:", end="")
    print(mape2, mae2, rmse2)
    print("KNN:", end="")
    print(mape3, mae3, rmse3)
    print("GP:", end="")
    print(mape4, mae4, rmse4)
    plt.figure()
    plt.plot(np.arange(len(truevalue)), truevalue, 'g', label='True value')
    plt.plot(np.arange(len(truevalue)), result2, 'r', linestyle='--', label='Random Forest Regression')
    plt.plot(np.arange(len(truevalue)), result1, 'y', linestyle='--', label='Support Vector Regression')
    plt.plot(np.arange(len(truevalue)), result3, 'b', linestyle='--', label='K-Nearest Neighbors Regression')
    plt.fill(np.concatenate([np.arange(len(truevalue)), np.arange(len(result4))[::-1]]),
             np.concatenate([result4 + 1.96 * err,
                             (result4 - 1.96 * err)[::-1]]),
             alpha=.5, fc='Silver', ec='None', label='95% confidence interval')
    plt.plot(np.arange(len(truevalue)), result4, 'black', linestyle='--',label='Gaussian Process Regression')

    plt.title("Comparison with several baseline methods",fontsize=15)
    font1={
        'family':'Times New Roman',
        'weight':'normal',
        'size':12,
    }
    plt.legend(prop=font1)
    plt.show()
if __name__ == '__main__':
    dir = r'D:/transfer_finally/data/'
    id = 4
    window = 60
    step = 5
    mode=1
    # main(dir,id,window,step,mode)
    com_figure(dir,id,window,step)
