# 第二次作业 SVM

——18373528 杨凌华

## 一、运行方法

打开MySVMwithSMO.py，将其中的TEST_FILE、OUTPUT_FILE修改为指定的文件名，然后点击运行该main函数即可。

## 二、各部分介绍

```python
# 用到的库
import numpy as np # 矩阵运算
import random # 随机数生成
import pandas as pd # csv文件读取和写入 
import pickle # 训练好的模型数据保存
```

### 1）训练数据加载

```python
def loadTrainDataSet(fileName):
    train = pd.read_csv(fileName).head(300)
    train = preProcess(train.copy())
    dataMat = np.array(train.loc[:, ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12']]).tolist()
    labelMat = np.array(train.loc[:,'label']).tolist()
    print(dataMat)
    print(labelMat)
    return dataMat, labelMat
```

利用pandas库读取csv文件，并将X和label分开，分别以列表的形式存到dataMat、labelMat中

### 2）SMO算法实现

```python
'''
dataMatIn: 存放X的二维列表
classLabels: 存放标签的一维列表
C: 惩罚参数
toler: 容错率
maxIter: 最大迭代次数
'''
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    # 利用将列表转化为矩阵
    dataMatrix = np.mat(dataMatIn) # m * n
    labelMatrix = np.mat(classLabels).transpose() # 1 * m --> m * 1
    # 获得输入特征矩阵的尺寸
    m,n = np.shape(dataMatrix)
    b = 0 # 初始化截距参数b
    iter = 0 # 初始化迭代次数
    # 初始化alphas矩阵
    alphas = np.mat(np.zeros((m,1))) # m * 1
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            # 计算误差Ei
            w = np.multiply(alphas, labelMatrix).T * dataMatrix # w = ∑αiyiXi  1 * n
            Fxi = w * dataMatrix[i,:].T + b # Fxi = w * Xi + b   // w:1*n Xi:n*1
            Ei = Fxi - float(labelMatrix[i])
            # 优化αi，并容许一定的容错率
            if (((labelMatrix[i]*Ei<-toler) and (alphas[i]<C)) or (((labelMatrix[i]*Ei)>toler) and (labelMatrix[i]>0))):
                j = selectJrand(i,m) # 随机获取一个除αi以外的αj，与αi组成一对进行优化
                # 计算误差Ej
                Fxj = w * dataMatrix[j,:].T + b
                Ej = Fxj - float(labelMatrix[j]) # 求j的误差
                # 提前保存更新之前的alpha
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                # 计算上下界H和L
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[j] + alphas[i])
                if L == H : print("L == H"); continue
                # 计算eta
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0 : print("eta >= 0"); continue
                # 更新αj
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                # 修剪αj
                alphas[j] = clipAlpha(alphas[j],H,L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                # 更新αi，与αj改变的大小相同，方向相反
                alphas[i] += labelMatrix[i] * labelMatrix[j] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                # 更新b
                if 0 < alphas[i] < C : b = b1
                elif 0 < alphas[j] < C : b = b2
                else : b = (b1 + b2) / 2.0
                # 统计优化次数
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d times" % (iter, i, alphaPairsChanged))
        # 更新迭代次数
        if alphaPairsChanged == 0 : iter += 1
        else : iter = 0
        print("iteration number: %d" % iter)
    # 获取优化后最终得到的系数矩阵
    w = np.multiply(alphas, labelMatrix).T * dataMatrix
    return b,alphas,w
```

### 3）随机获取αj

```python
def selectJrand(i,m):
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j
```

### 4）根据上下界裁剪αj

```python
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj
```

