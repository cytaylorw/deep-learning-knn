import numpy as np
import matplotlib.pylab as plt
import csv
import tensorflow as tf


label0 = 'setosa'
label1 = 'versicolor'
label2 = 'virginica'
plt.rcParams['figure.figsize'] = (20.0, 10.0)

def PlotData(data, label, printLabel=False):
    if printLabel:
        print(label)
        print(len(label))
    class0 = class1 = class2 = False
    if len(data) == len(label):
        for i in range(len(label)):
            if label[i] == 0:
                if class0:
                    plt.plot( data[i,0], data[i,1], 'ro')
                else:
                    plt.plot( data[i,0], data[i,1], 'ro', label=label0)
                    class0 = True
            elif label[i] == 1:
                if class1:
                    plt.plot( data[i,0], data[i,1], 'go')
                else:
                    plt.plot( data[i,0], data[i,1], 'go', label=label1)
                    class1 = True
            elif label[i] == 2:
                if class2:
                    plt.plot( data[i,0], data[i,1], 'bo')
                else:
                    plt.plot( data[i,0], data[i,1], 'bo', label=label2)
                    class2 = True
                
    

# Function 讀data進來
def ReadIrisDataFromCsv(filename):
    '''自CSV檔讀取Iris資料
    Args:
        filename (string): CSV檔路徑
    Return:
        Iris資料
    '''

    # 開啟 CSV 檔案
    with open(filename, newline='') as csvfile:

        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile, delimiter=',')
        allDataList = list(rows)
        allData = np.asarray(allDataList)

        # 取特徵與標籤
        data = (allData[1:,0:2]).astype(np.float)
        labelStr = allData[1:,4]
        label = np.select([labelStr=='setosa',labelStr=='versicolor',labelStr=='virginica'], [0,1,2])
        ##ZPlotData(data,label,True)
        #plt.legend(loc='best')
        plt.show()

    return (data, label)
	
def CreateParameters(X_t, y_t, x_t, k_t):
    '''建立計算用的變數
        Args:
            ...: 代表所需的引入參數，請自行設計
    Return:
        計算用的變數
	Remark:
		使用tf.nn.top_k(...) 這個function會比較好做
    '''
    neg_one = tf.constant(-1.0, dtype=tf.float64)
    # we compute the L-1 distance
    distances =  tf.reduce_sum(tf.abs(tf.subtract(X_t, x_t)), 1)
    # to find the nearest points, we find the farthest points based on negative distances
    # we need this trick because tensorflow has top_k api and no closest_k or reverse=True api
    neg_distances = tf.multiply(distances, neg_one)
    # get the indices
    vals, indx = tf.nn.top_k(neg_distances, k_t)
    # slice the labels of these points
    y_s = tf.gather(y_t, indx)
    return y_s

# 主程式段落開始：
print('Start KNN Test...')


# 全部的測試點
testData = np.array([[5.4, 3.2],
                     [7, 3.3],
                     [6.1, 3.2]], dtype='float64')

(data, label) = ReadIrisDataFromCsv('./Iris.csv')

# 測試用的K值
TestK = [1, 2, 3, 5, 7, 9]

print('Online Test...') #Online測試階段
for test in testData:
    # 印出Feature Map，包含  testing 與 training 資料點
    PlotData(data,label)
    plt.plot(test[0], test[1], 'y', marker='D', markersize=10, label='test point')
    plt.legend(loc='best')
    plt.show()
    print('Test data: {0}'.format(test))
    # KNN的K，這裡記得用迴圈取出不同的K
    for k in TestK:
        print('K = {0}'.format(k))
        k_tf = tf.constant(k) # 挑TestK中的幾種來做測試，或是全做來比較結果。
        sess = tf.Session()
        pr = CreateParameters(data, label, test, k_tf)
        index = sess.run(pr)

        # 印出K個最近的樣本所屬 類別名稱或index
        print('\t{0}'.format(index))
        # 顯示最後辨識結果 類別名稱    
        result = np.argmax(np.bincount(index.astype('int64')))
        print('\t{0}: {1}'.format(result,np.select([result==0,result==1,result==2], [label0,label1,label2])))
        
    
    