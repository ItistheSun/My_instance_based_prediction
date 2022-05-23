import csv
import numpy as np
import math
from sklearn.cluster import KMeans

linkID = {801230, 801241, 801255, 801288, 801595, 805949, 806574, 807818, 808292, 808993, 809117, 809188, 809294,
          809485, 810131, 810210, 813316, 813973, 814211, 816299, 816400, 816464, 818157, 818549, 818553, 818609,
          818619, 818723, 819484, 819684, 819703, 819728, 821646, 822504, 823245, 824044, 824105, 824166, 825334,
          825818, 825887, 825899, 826013, 827111, 827152, 828491}
date=['01','02','03','04','05','06']
class link():
    def __init__(self,id,f):
        self.id=id
        self.f=f
def read_data_avgflow(dir):
    minu = 12
    data = []
    for id in linkID:
        linkdata = []
        for m in range(len(date)):
            d = date[m]
            path = dir+'Station_data_history/link%s_%s.csv' % (id, d)
            with open(path) as f:
                f_csv = csv.reader(f)
                for row in f_csv:
                    for i in range(len(row)):
                        row[i] = float(row[i])
                    linkdata.append(row[1:])
        his = []
        m = 0
        n = m + minu
        while n <= len(linkdata):
            temp = linkdata[m:n]
            temp = np.array(temp)
            temp = np.sum(temp, axis=0)
            his.append(temp[0])
            m = n
            n = m + minu
        his = np.array(his)
        y=[]
        y.append(his.mean())
        y=np.array(y)
        data.append(link(id, y))
    return np.array(data)
def read_data_flow_timeslice(dir):
    minu = 12
    data = []
    for id in linkID:
        linkdata = []
        for m in range(len(date)):
            d = date[m]
            path = dir+'Station_data_history/link%s_%s.csv' % (id, d)
            with open(path) as f:
                f_csv = csv.reader(f)
                for row in f_csv:
                    for i in range(len(row)):
                        row[i] = float(row[i])
                    linkdata.append(row[1:])
        his = []
        m = 0
        n = m + minu
        while n <= len(linkdata):
            temp = linkdata[m:n]
            temp = np.array(temp)
            temp = np.sum(temp, axis=0)
            his.append(temp[0])
            m = n
            n = m + minu
        his = np.array(his)
        y=np.array(his)
        data.append(link(id, y))
    return np.array(data)

def get_disance(a,b):
    a=np.array(a)
    b=np.array(b)
    distance=np.sqrt(np.sum((a-b)*(a-b)))
    return distance

def test_Kmeans(data,k,r):
    datas = []
    links = []
    lenf=(data[0].f).shape[0]
    for i in range(len(data)):
        datas.append(data[i].f)
        links.append(data[i].id)
    datas = np.array(datas).reshape(-1, lenf)
    estimator = KMeans(n_clusters=k, random_state=r)
    estimator.fit(datas)
    labels = estimator.labels_
    centroids=estimator.cluster_centers_
    return centroids, labels,links


def get_cluster(data,k,r):
    centroids, clusterAssment,links = test_Kmeans(data, k,r)
    s=[]
    p = []
    for i in range(k):
        distance=[]
        l = np.nonzero(clusterAssment == i)[0]
        list = set()
        for j in l:
            list=list.union({data[j].id})
            distance.append(get_disance(centroids[i],data[j].f))
        distance=np.array(distance)
        s.append(distance.mean())
        p.append(list)
    DBI=0
    for i in range(k):
        max=0
        for j in range(k):
            if i!=j:
                m = get_disance(centroids[i], centroids[j])
                temp = (s[i] + s[j]) / m
                if temp > max:
                    max = temp
        DBI=DBI+max
    DBI=DBI/k
    weight=math.exp(DBI**-1)
    return weight,p

def get_result_new(linkt,k):
    dir=r'D:/transfer_finally/PeMS/'
    data1 = read_data_avgflow(dir)
    data3=read_data_flow_timeslice(dir)
    w1, p1 = get_cluster(data1, k, 1)
    print("聚类1：")
    print(p1)
    print(w1)
    w2, p2 = get_cluster(data3, k, 0)
    print("聚类2：")
    print(p2)
    print(w2)
    p = []
    p.append(p1)
    p.append(p2)
    cluster_p1={}
    cluster_p2={}
    for c in p1:
        for l in c:
            temp=list(c)
            temp.remove(l)
            cluster_p1[str(l)]=temp
    for c in p2:
        for l in c:
            temp=list(c)
            temp.remove(l)
            cluster_p2[str(l)]=temp
    print(cluster_p1)
    print(cluster_p2)
    simlink=[]
    simset=[]
    for id in linkt:
        s1=cluster_p1[str(id)]
        s2 = cluster_p2[str(id)]
        st=set(s1).intersection(set(s2))
        if len(st)==0:
            print(id)
            if w1>w2:
                st=s1
            else:
                st=s2
        fs=[]
        for m in st:
            if m not in linkt:
                fs.append(m)
        simlink.append(id)
        simset.append(fs)
    print(simlink)
    print(simset)
    return simlink,simset
