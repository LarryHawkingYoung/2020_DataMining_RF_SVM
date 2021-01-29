class MyTree:
    def __init__(self):
        self.left = None            # 左子树
        self.right = None           # 右子树
        self.split_feature = None   # 划分特征名
        self.split_value = None     # 划分特征值
        self.value = None           # 叶节点的值  只有叶节点有

    def predict(self, sample, features):    # 多个样本
        if self.value is not None:      # 叶节点
            return self.value
        else:
            feature_num = features.index(self.split_feature)

            if sample[feature_num] >= self.split_value:
                if self.left is not None:
                    return self.left.predict(sample,features)
                else:
                    return self.right.predict(sample,features)
            else:
                if self.right is not None:
                    return self.right.predict(sample,features)
                else:
                    return self.left.predict(sample,features)
