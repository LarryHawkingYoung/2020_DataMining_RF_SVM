# 第一次作业	Random Forest

——18373528杨凌华

## 一、运行方法

将所有.py、.csv以及我自定义的模型存储文件.larry放在同一目录下，打开MyRandomForestClassifier.py，将TEST_FILE和OUTPUT_FILE分别修改为要求的测试文件名和预测输出文件名，宏观变量USE_TRAINED_MODEL和USE_NEW_TESTSET都置为true，然后运行该文件，即可迅速将预测结果输出到指定文件中。

## 二、各部分介绍

整体分为四个.py文件

### 1）MyRandomForestClassifier.py

这是存放main函数的文件。

是项目运行的起始文件，它用于读取数据集文件，进行数据预处理，并进行模型训练和预测输出。

### 2）MyTree.py

决策树子树结点类。

包含每一棵决策树子树结点的信息：

![image-20201223223558327](C:\Users\yang\AppData\Roaming\Typora\typora-user-images\image-20201223223558327.png)

同时，具有一个预测函数predict()，通过传入某一个样本，根据该树节点的划分特征名以及划分特征值，来调用左右子树的predict进行更深一层的决策判断，知道遇到叶节点return叶节点的value。

### 3）Data.py

数据结构的封装。

它包含了一组样本的所有信息，将预处理后的数据封装成Data类，之后送去进行训练。

![image-20201223223951696](C:\Users\yang\AppData\Roaming\Typora\typora-user-images\image-20201223223951696.png)

### 4）MyRandomForest.py

用于模型训练与预测的主要源文件。

![image-20201223224524340](C:\Users\yang\AppData\Roaming\Typora\typora-user-images\image-20201223224524340.png)

### 5）fit函数

用于模型的训练，首先随机选取 n_tree 组特征，建立 n_tree 颗决策树，组成一个随机森林

```python
    def fit(self):
        # 构建 n_tree 棵树
        for i in range(self.n_tree):

            new_data = []                     # 所选样本数据  二维列表
            new_feature = []                  # 所选特征的名字  列表
            new_feature_num = []              # 所选特征的列号  列表

            ###################### 随机选取特征开始 ##################################
            for j in range(self.n_features):  # 随机选取的特征  随机选择 n_features 个特征
                # randint是闭区间
                col = random.randint(0, len(self.data.data[0]) - 2) # 在 x1~x22 的所有特征中选一个的列号
                new_feature_num.append(col)
                new_feature.append(self.data.features[col])
            new_feature.append(self.target) # 把目标特征label名字加到末尾
            new_feature_num.append(len(self.data.features) - 1) # 把目标特征的编号加到末尾
            ###################### 随机选取特征完毕 ##################################

            ###################### 随机选取样本数据开始 ###############################
            for j in range(self.nums):  # 随机选取的样本数据  随机选择 nums 个特征
                new_sample = []
                # 随机选取一行样本
                row = random.randint(0, self.nums - 1)
                for col in new_feature_num:     # 依次提取出抽中的样本的数据
                    new_sample.append(self.data.get_larry_value(row,col))
                new_data.append(new_sample)
            ###################### 随机选取样本数据完毕 ###############################

            new_data = Data(new_feature, self.target, new_data)
            tree_importance = {}
            # 以 0 为深度起始值，递归建树
            self.trees.append(self.build_tree(new_data, self.target, 0, tree_importance))  # 建树
            self.importance.append(tree_importance)
            print("第" + str(i + 1) + "棵树学习完成")

        for tree_importance in self.importance:
            for key in tree_importance.keys():
                value = tree_importance[key]
                if key in self.forest_importance_:
                    self.forest_importance_[key] = self.forest_importance_[key] + value # 重要性直接相加
                else:
                    self.forest_importance_[key] = value
```

### 6）建树

以平方平均误差作为指标，选取该指标最小的特征作为划分特征，将当前树分为两棵子树

```python
    def build_tree(self, data, target, depth, importance):
        if self.min_to_split >= data.nums or depth >= self.max_depth:
            if data.nums == 0:
                return None
            tree = MyTree()
            value_list = data.get_data(target)
            tree.value = sum(value_list) / len(value_list) # 取平均值
            return tree
        else:
            best_feature, best_value = self.split(data)
            # if best_feature is None:
            #     tree = MyTree()
            #     value_list = data.get_data(target)
            #     tree.value = sum(value_list) / len(value_list)
            #     return tree

            col = data.features.index(best_feature)
            tree = MyTree()
            left_data = []
            right_data = []
            for row in data.data:
                if row[col] >= best_value:
                    row.pop(col)
                    left_data.append(row)
                else:
                    row.pop(col)
                    right_data.append(row)
            tree.split_feature = best_feature
            importance[tree.split_feature] = self.max_depth - depth # 深度越大 重要性越小
            tree.split_value = best_value
            data.features.remove(best_feature)
            # python 函数直接传递的引用而非形参数值
            left_data = Data(data.features, target, left_data)
            right_data = Data(data.features[:], target, right_data) # [:] 复制一份
            tree.left = self.build_tree(left_data, target, depth + 1, importance)
            tree.right = self.build_tree(right_data, target, depth + 1, importance)
            return tree
        
        
    def split(self, data):
        best_feature = None
        best_value = None
        best_gain = None
        for feature in data.features:
            if feature != self.target:
                for value in data.get_unique_data(feature):
                    gain = self.cal_gain(data, feature, value)
                    if best_gain is None:
                        best_gain = gain
                        best_feature = feature
                        best_value = value
                    else:
                        if gain < best_gain: # 越小越好
                            best_gain = gain
                            best_feature = feature
                            best_value = value
        return best_feature, best_value
    
    
    def cal_gain(self, data, feature, value):     # 平方平均误差(方差) 越小越好
        left_data = [] # >= value
        right_data = [] # < value
        col = data.features.index(feature)
        for row in data.data:
            if row[col] >= value:
                left_data.append(row)
            else:
                right_data.append(row)
        # 求 >=value 数据的平均值
        left_sum = 0
        left_avg = 0
        for row in left_data:
            left_sum += row[col]
        if len(left_data) != 0:
            left_avg = left_sum / len(left_data)
        # 求 <value 数据的平均值
        right_sum = 0
        right_avg = 0
        for row in right_data:
            right_sum += row[col]
        if len(right_data) != 0:
            right_avg = right_sum / len(right_data)

        sum_error = 0
        for row in left_data:
            sum_error += (row[col] - left_avg) ** 2
        for row in right_data:
            sum_error += (row[col] - right_avg) ** 2
        return sum_error
```

