import numpy as np
import random
import pandas as pd
import pickle

def selectJrand(i,m):
    j = i
    while j == i:
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
        aj = L
    return aj

'''
dataMatIn: 存放X的二维列表
classLabels: 存放标签的一维列表
C: 惩罚参数
toler: 容错率
maxIter: 最大迭代次数
'''
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    dataMatrix = np.mat(dataMatIn) # m * n
    labelMatrix = np.mat(classLabels).transpose() # 1 * m --> m * 1
    m,n = np.shape(dataMatrix)
    b = 0
    iter = 0
    alphas = np.mat(np.zeros((m,1))) # m * 1
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            w = np.multiply(alphas, labelMatrix).T * dataMatrix # w = ∑αiyiXi  1 * n
            Fxi = w * dataMatrix[i,:].T + b # Fxi = w * Xi + b   // w:1*n Xi:n*1
            Ei = Fxi - float(labelMatrix[i]) # 计算误差
            if (((labelMatrix[i]*Ei<-toler) and (alphas[i]<C)) or (((labelMatrix[i]*Ei)>toler) and (labelMatrix[i]>0))):
                j = selectJrand(i,m)
                Fxj = w * dataMatrix[j,:].T + b
                Ej = Fxj - float(labelMatrix[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMatrix[i] != labelMatrix[j]:
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[j] + alphas[i])
                if L == H : print("L == H"); continue
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0 : print("eta >= 0"); continue
                alphas[j] -= labelMatrix[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += labelMatrix[i] * labelMatrix[j] * (alphaJold - alphas[j])
                b1 = b - Ei - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMatrix[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMatrix[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                if 0 < alphas[i] < C : b = b1
                elif 0 < alphas[j] < C : b = b2
                else : b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d times" % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0 : iter += 1
        else : iter = 0
        print("iteration number: %d" % iter)
    w = np.multiply(alphas, labelMatrix).T * dataMatrix
    return b,alphas,w

def loadTrainDataSet(fileName):
    train = pd.read_csv(fileName).head(200)
    train = preProcess(train.copy())
    dataMat = np.array(train.loc[:, ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12']]).tolist()
    labelMat = np.array(train.loc[:,'label']).tolist()
    print(dataMat)
    print(labelMat)
    return dataMat, labelMat

def loadTestDataSet(fileName):
    test = pd.read_csv(fileName)
    test = preProcess(test.copy())
    dataMat = np.array(
        test.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']]).tolist()
    labelMat = np.array(test.loc[:, 'label']).tolist()
    return np.mat(dataMat), np.mat(labelMat).transpose()

def loadTestInClass(fileName):
    test = pd.read_csv(fileName)
    test = preProcess(test.copy())
    dataMat = np.array(
        test.loc[:, ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12']]).tolist()
    indices = np.array(test.loc[:, 'index']).tolist()
    return np.mat(dataMat), indices

def preProcess(df):
    a = 0.50
    b = 0.50
    df['x1'] = a * (df['x1'] / df['x1'].sum()) * 10.0 + (df['x1'] / df['x1'].max()) * 10.0 * b
    df['x2'] = a * (df['x2'] / df['x2'].sum()) * 10.0 + (df['x2'] / df['x2'].max()) * 10.0 * b
    df['x3'] = a * (df['x3'] / df['x3'].sum()) * 10.0 + (df['x3'] / df['x3'].max()) * 10.0 * b
    df['x4'] = a * (df['x4'] / df['x4'].sum()) * 10.0 + (df['x4'] / df['x4'].max()) * 10.0 * b
    df['x5'] = a * (df['x5'] / df['x5'].sum()) * 10.0 + (df['x5'] / df['x5'].max()) * 10.0 * b
    df['x6'] = a * (df['x6'] / df['x6'].sum()) * 10.0 + (df['x6'] / df['x6'].max()) * 10.0 * b
    df['x7'] = a * (df['x7'] / df['x7'].sum()) * 10.0 + (df['x7'] / df['x7'].max()) * 10.0 * b
    df['x8'] = a * (df['x8'] / df['x8'].sum()) * 10.0 + (df['x8'] / df['x8'].max()) * 10.0 * b
    df['x9'] = a * (df['x9'] / df['x9'].sum()) * 10.0 + (df['x9'] / df['x9'].max()) * 10.0 * b
    df['x10'] = a * (df['x10'] / df['x10'].sum()) * 10.0 + (df['x10'] / df['x10'].max()) * 10.0 * b
    df['x11'] = a * (df['x11'] / df['x11'].sum()) * 10.0 + (df['x11'] / df['x11'].max()) * 10.0 * b
    df['x12'] = a * (df['x12'] / df['x12'].sum()) * 10.0 + (df['x12'] / df['x12'].max()) * 10.0 * b
    return df

def saveModelW(w):
    model_file_w = open('trained_model_w.larry', 'wb')
    pickle.dump(w, model_file_w)
    model_file_w.close()

def saveModelB(b):
    model_file_b = open('trained_model_b.larry', 'wb')
    pickle.dump(b, model_file_b)
    model_file_b.close()

def saveModelAlphas(alphas):
    model_file_alphas = open('trained_model_alphas.larry', 'wb')
    pickle.dump(alphas, model_file_alphas)
    model_file_alphas.close()

def loadModelW():
    model_file_w = open('trained_model_w.larry', 'rb')
    w = pickle.load(model_file_w)
    model_file_w.close()
    return w

def loadModelB():
    model_file_b = open('trained_model_b.larry', 'rb')
    b = pickle.load(model_file_b)
    model_file_b.close()
    return b

def loadModelAlphas():
    model_file_alphas = open('trained_model_alphas.larry', 'rb')
    alphas = pickle.load(model_file_alphas)
    model_file_alphas.close()
    return alphas

if __name__ == '__main__':

    OUTPUT_FILE = 'predict.csv'
    TEST_FILE = 'test.csv'

    # dataMat, labelMat = loadTrainDataSet('svm_training_set.csv')

    # b, alphas, w = smoSimple(dataMat, labelMat, 0.6, 0.001, 30)

    alphas = loadModelAlphas()
    w = loadModelW()
    b = loadModelB()

    dataMat, indices = loadTestInClass(TEST_FILE)
    preds = []
    row, col = np.shape(dataMat)
    for i in range(row):
        y = w * dataMat[i, :].T + b
        if y > 0: preds.append(1)
        else: preds.append(-1)
    output = pd.DataFrame({'index': indices, 'label': preds})
    output.to_csv(OUTPUT_FILE, index=False)
