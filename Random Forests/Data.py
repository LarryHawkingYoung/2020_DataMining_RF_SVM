class Data:

    def __init__(self,features,target,origin_data):
        self.nums = len(origin_data)   # 样本数
        self.features = features       # 所有特征  一维列表，存储所有特征名字，包括目标特征，字符串
        self.target = target           # 目标特征  字符串，存储目标特征标签名字，字符串
        self.data = origin_data        # 数据      二维列表

    # 根据 row col 获取某一位置的数据
    def get_larry_value(self,row,col):
        return self.data[row][col]

    # 提取某一特征名下的所有值，组成一维列表
    def get_data(self,feature):
        ans = []
        col = self.features.index(feature)
        for row in range(self.nums):
            ans.append(self.data[row][col])
        return ans

    # 提取某一特征名下的所有值，利用集合set去重，组成一维列表
    def get_unique_data(self,feature):
        return list(set(self.get_data(feature)))

    def __str__(self):
        return "样本数： " + str(self.nums) + "\n样本特征: " + str(self.features) + "\n目标特征: " + str(self.target)
