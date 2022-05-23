import numpy as np
import csv
from sklearn import preprocessing
from sklearn import metrics
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern,WhiteKernel, ConstantKernel as C
import warnings
warnings.filterwarnings("ignore")
'''路网测试，有滑动窗口,五折交叉验证'''
linkt={825899,806574,825818,814211,818553,818619,824166,801595,819703,816400}
linkID = {801230, 801241, 801255, 801288, 801595, 805949, 806574, 807818, 808292, 808993, 809117, 809188, 809294,
          809485, 810131, 810210, 813316, 813973, 814211, 816299, 816400, 816464, 818157, 818549, 818553, 818609,
          818619, 818723, 819484, 819684, 819703, 819728, 821646, 822504, 823245, 824044, 824105, 824166, 825334,
          825818, 825887, 825899, 826013, 827111, 827152, 828491}

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
    for m in range(len(datet)):
        data = []
        d=datet[m]
        path = dir+'Station_data/link%s_%s.csv' % (id,d)
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
    x_data = np.hstack((preprocessing.scale(x_data[:, 0:2]), x_data[:, 2:3]))
    y_data=np.array(y_data)
    return x_data,y_data
def test_data(dir,d,id,window,step):
    data = []
    x_data=[]
    y_data=[]
    path = dir+'Station_data/link%s_%s.csv' % (id,d)
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
def main():
    f = open("Network_cross_pems.csv", 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["link_id", "mape", "mae", "rmse"])
    date = ["07", "08", "09", "10", "11"]
    dir = r'D:/transfer_finally/PeMS/'
    mapeall = []
    maeall = []
    rmseall = []
    links=linkID-linkt
    for id in links:
        window = 12
        step = 12
        mape5 = []
        mae5 = []
        rmse5 = []
        for i in date:
            datet = date[:]
            datet.remove(i)
            train_x, train_y = train_data(dir, datet, id, window, step)
            test_x, test_y = test_data(dir, i, id, window, window)
            kernel1 = C(constant_value=0.1, constant_value_bounds=(1e-10, 1e10)) * Matern(length_scale=0.5,
                                                                                          length_scale_bounds=(
                                                                                              1e-10, 1e10),
                                                                                          nu=1.5)+WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e5))
            gp = GaussianProcessRegressor(kernel=kernel1, normalize_y=True, n_restarts_optimizer=50, random_state=123)
            gp.fit(train_x, train_y)
            resultrf, sigma = gp.predict(test_x, return_std=True)
            mape1 = getMAPE(test_y, resultrf)
            mae1 = getMAE(test_y, resultrf)
            rmse1 = getRMSE(test_y, resultrf)
            mape5.append(mape1)
            mae5.append(mae1)
            rmse5.append(rmse1)
        mape = np.mean(mape5)
        mae = np.mean(mae5)
        rmse = np.mean(rmse5)
        mapeall.append(mape)
        maeall.append(mae)
        rmseall.append(rmse)
        csv_writer.writerow([id, mape, mae, rmse])
        print("link%s:" % id, end="")
        print(mape, mae, rmse)
    average_mape = np.mean(mapeall)
    average_mae = np.mean(maeall)
    average_rmse = np.mean(rmseall)
    print("mape", average_mape)
    print("mae", average_mae)
    print("rmse", average_rmse)
    csv_writer.writerow(["average", average_mape, average_mae, average_rmse])
    f.close()
if __name__ == '__main__':
    main()