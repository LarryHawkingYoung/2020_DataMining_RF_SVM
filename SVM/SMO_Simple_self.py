import numpy as np
import random
import pandas as pd
import pickle

class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  # Initialize the structure with the parameters
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = np.shape(dataMatIn)[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2))) #first column is valid flag
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)

def kernelTrans(X, A, kTup): #calc the kernel or transform data to a higher dimensional space
    m,n = np.shape(X)
    K = np.mat(np.zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #linear kernel
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #divide in NumPy is element-wise not matrix like Matlab
    else: raise NameError('Houston We Have a Problem -- \
    That Kernel is not recognized')
    return K

def calcEk(oS, k):
    fXk = float(np.multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = np.nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue  # don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy()
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #added this for the Ecache
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(np.mat(dataMatIn),np.mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   #go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#go over non-bound (railed) alphas
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #toggle entire set loop
        elif (alphaPairsChanged == 0): entireSet = True
        print("iteration number: %d" % iter)
    w = np.multiply(oS.alphas, oS.labelMat).T * oS.X
    return oS.b,oS.alphas,w




def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

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
    train = pd.read_csv(fileName).head(1000)
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

def preProcess(df):
    a = 0.70
    b = 0.30
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
    # df['x2'] = df['x2'] / 10.0
    # df['x3'] = df['x3'] / 100000.0
    # df['x10'][df['x10'] > 0] = 1
    # df['x11'][df['x11'] > 0] = 1
    # df['x5'] = df['x5'] - df['x5'].min()
    # df['x11'] = df['x11'] / df['x11'].mean()
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
    dataMat, labelMat = loadTrainDataSet('svm_training_set.csv')
    # dataMat, labelMat = loadDataSet('testSetSelf.txt')

    # b, alphas, w = smoSimple(dataMat, labelMat, 0.6, 0.0001, 40)

    b, alphas, w = smoP(dataMat, labelMat, 0.6, 0.001, 40)

    # saveModelAlphas(alphas)
    # saveModelW(w)
    # saveModelB(b)

    # alphas = loadModelAlphas()
    # w = loadModelW()
    # b = loadModelB()

    print(w)
    print(b)
    print(alphas[alphas>0])

    dataMat, labelMat = loadTestDataSet('svm_training_set.csv')
    row, col = np.shape(dataMat)
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(row):
        y = w * dataMat[i,:].T + b
        print("y:\t" + str(y) + "\tlabel:\t" + str(labelMat[i]))
        if y > 0:
            if labelMat[i] > 0: TP += 1
            else: FP += 1
        else:
            if labelMat[i] > 0: FN += 1
            else: TN += 1
    print("TP:\t" + str(TP))
    print("FP:\t" + str(FP))
    print("TN:\t" + str(TN))
    print("FN:\t" + str(FN))
    precision = float(TP) / (TP + FP)
    recall = float(TP) / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    print("precision:\t" + str(precision))
    print("recall:\t" + str(recall))
    print("F1:\t" + str(F1))