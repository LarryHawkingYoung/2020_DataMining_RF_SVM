import pandas
from sklearn.model_selection import train_test_split
# from sklearn.externals import joblib
import pickle
import numpy as np
from Data import Data
import random
from MyRandomForest import MyRandomForest
import matplotlib.pyplot as plt

if __name__ == '__main__':
    USE_TRAINED_MODEL = True # 用以训练好的模型
    USE_NEW_TESTSET = True # 用新的测试库
    TEST_FILE = 'test.csv'
    OUTPUT_FILE = 'predict.csv'

    print("-----读入数据开始-----")
    df1 = pandas.read_csv("x_train.csv")
    df2 = pandas.read_csv("y_train.csv")
    print("-----读入数据完毕-----")

    print("-----数据预处理开始-----")
    df = pandas.merge(df1, df2, on='index')
    x = df.loc[:, ['index', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
                      'x8', 'x9', 'x10', 'x11', 'x12', 'x13',
                      'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                      'x21', 'x22']]
    y = df.loc[:, ['label']]
    print("-----数据预处理完毕-----")

    print("-----训练集与测试集划分开始-----")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)
    print("-----训练集与测试集划分完毕-----")

    tmp = np.array(x_test.loc[:,['index']]).tolist()
    indices = []
    for t in tmp:
        indices.append(t[0])
    x_train = x_train.drop('index',axis=1)
    x_test = x_test.drop('index',axis=1)

    print("-----数据提取与整合开始-----")
    # 训练集
    train_data = np.array(x_train).tolist()
    train_data_result = np.array(y_train).tolist()
    # 测试集
    test_data = np.array(x_test).tolist()
    test_data_result = np.array(y_test).tolist()

    for i in range(len(train_data)):
        train_data[i].append(train_data_result[i][0])
    for i in range(len(test_data)):
        test_data[i].append(test_data_result[i][0])

    features = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7',
                      'x8', 'x9', 'x10', 'x11', 'x12', 'x13',
                      'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20',
                      'x21', 'x22', 'label']

    train_data = Data(features, 'label', train_data)
    test_data = Data(features, 'label', test_data)
    print("-----数据提取与整合完毕-----")

    print("-----训练开始-----")
    # (Data类训练数据，森林中决策树个数，决策树最大深度，随机选取的特征数，最小划分数量)
    if USE_TRAINED_MODEL:
        print("-----加载已经训练好的模型-----")
        model_file = open('trained_model.larry','rb')
        random_forest = pickle.load(model_file)
        model_file.close()
    else:
        print("-----重新训练新的模型-----")
        random_forest = MyRandomForest(train_data, 20, 12, 17, 3)
        random_forest.fit()
        model_file = open('trained_model.larry', 'wb')
        pickle.dump(random_forest,model_file)
        model_file.close()
    print("-----训练完毕-----")

    print("-----预测开始-----")
    if USE_NEW_TESTSET:
        df = pandas.read_csv(TEST_FILE)
        df['label'] = 0
        # print(df)
        indices = np.array(df.loc[:, 'index']).tolist()
        df = df.drop('index', axis=1)
        test_data = np.array(df).tolist()
        test_data = Data(features, 'label', test_data)
        # print(test_data)
        # print(test_data.data)
        predict_data = random_forest.predict_test(test_data)
    else: predict_data = random_forest.predict(test_data)
    output = pandas.DataFrame({'index':indices,'label':predict_data})
    output.to_csv(OUTPUT_FILE,index=False)
    print("-----预测完毕-----")

    importance = random_forest.get_importance()
    x = []
    y = []
    for key in importance:
        x.append(key)
        y.append(importance[key])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family'] = 'sans-serif'
    plt.bar(x, y)
    plt.show()
