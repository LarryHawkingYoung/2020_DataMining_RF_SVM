from MyTree import MyTree
from Data import Data
import random

class MyRandomForest:
    def __init__(self, data, n_tree, max_depth, n_features, min_to_split):
        # 直接初始化参数
        self.data = data                    # 原始数据  Data类的对象
        self.n_tree = n_tree                # 决策树的数量
        self.max_depth = max_depth          # 最大深度
        self.n_features = n_features        # 随机选取的特征数
        self.min_to_split = min_to_split    # 最小划分数量
        # 间接初始化参数
        self.target = data.target           # 目标特征名  字符串
        self.nums = data.nums               # 原始样本数量
        # 训练过程中生成的模型数据
        self.trees = []                     # 训练后生成的树  列表
        self.importance = []                # 存放每一棵树的 各个特征重要性记录 是一个集合的数组
        self.forest_importance_ = {}        # 存放所有树综合后 各个特征的重要性记录

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

    def predict_test(self,test_data):
        predict_data = []
        for sample in test_data.data:
            sum = 0
            for tree in self.trees:
                sum += tree.predict(sample,test_data.features)
            avg = sum / self.n_tree
            # print("avg:\t" + str(avg))
            if avg > 0: predict_data.append(1)
            else: predict_data.append(-1)
        return predict_data

    def predict(self,test_data):
        predict_data = []
        origin_data = []
        right = 0
        wrong = 0
        for sample in test_data.data:
            sum = 0
            for tree in self.trees:
                sum += tree.predict(sample,test_data.features)
            avg = sum / self.n_tree
            feature_num = test_data.features.index(self.target)
            if sample[feature_num] == -1 and avg < 0 or sample[feature_num] == 1 and avg > 0:
                right += 1
            else:
                # print("right:\t" + str(sample[feature_num]) + "\twrong:\t" + str(avg))
                wrong += 1
            # print("avg:\t" + str(avg))
            if avg > 0: predict_data.append(1)
            else: predict_data.append(-1)

            origin_data.append(sample[feature_num])
        print("right: " + str(right))
        print("wrong: " + str(wrong))
        print("准确率：",right / (right + wrong),sep="  ")
        print(" ")
        return predict_data

    def get_importance(self):
        ans = {}
        for i in sorted(self.forest_importance_.items(), key=lambda kv: (kv[1], kv[0])):
            ans[i[0]] = i[1]
        sum = 0
        for key in ans:
            sum += ans[key]
        for key in ans:
            ans[key] = ans[key] / sum # 求重要性占比
        return ans
