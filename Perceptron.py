import numpy as np

class Perceptron:

    def __init__(self, learningRate = 0.02, number_iter = 10):
    # _init_ 初始化函数, learingRate 和 训练数据set的个数

    # learningRate 
    # number_iter 表示进行 训练次数
        self.learningRate = learningRate;
        self.number_iter = number_iter;
    
    # 初始化新的对象，使用data
    # name_ 表示不在 _init_中初始化的属性，只在使用其他函数的时候才创建

    def fit(self, X, y):
    # w_[] list 表示适配之后的权重weights 表示不同的featureNumber 对应的权重
    # errors_[] lsit 表示在每次迭代中epoch中 的分类错误的个数

        ### X 表示一个numpy的二维矩阵 [sampleNumber, featuresNumber]
        ### y 表示一个numpy的一维矩阵，表示[sampleNumber] ==> 每一个sample的真是的值？ -1 or +1
        ### [a, b] ==> 对应相应的值
        ### X.shape ==> 返回每一个维度的长度，(n, n)
        self.w_ = np.zeros(1 + X.shape[1]); # 表示 feature维度的长度，然后加上1，表示threshold, 全是0
        #### 一维的数组

        self.errors_ = [];

        for _ in range(self.number_iter):
            errors = 0;  #### 每一次epoch的错误个数

            for xi, target in zip(X, y):   #### zip 之后对应的 ( [value1, value2, value3], sampleNumber) sampleNumber 对应的features vector

                ### 每次epoch 都会修改模型，然后再次预测
                update = self.learningRate * (target - self.predict(xi));

                #### update 对应的是此时这个样本 sample对应的结果
                #### xi 对应的是 [1,2,3,3,3] 是每一个weight 对应的 features值
                #### 此时要更新每一个weight ， 通过 += update * [1,2,3,4,4]


                self.w_[1:] += update * xi; #### 矩阵后面对应feachersNumber 的weight 要变化 初始的时候是0
                self.w_[0] += update; #### 对应的阈值也要变化 初始的时候是0  这里根据的是update的原则，注意阈值对应的xi是1

                #### w0 对应 x0是1，所以w0就表示的是阈值，所以输入的时候不用输入x0 无意义

                errors += int(update != 0.0);  #### 如果update不是0，表示不正确预测
            
            #### 将每一次遍历的 错误个数存入list中
            self.errors_.append(errors);

        return self;

    #### xi 表示单个sanple下的，表示对应的feachersNuber 一维 vector
    def net_input(self, xi):
        return np.dot(xi, self.w_[1:]) + self.w_[0]; ### 注意加上阈值

    # 进行预测
    #### xi 表示单个sanple下的 一维 vector
    def predict(self, xi):
        return np.where(self.net_input(xi) >= 0.0, 1, -1);


    #### 注意 每一次epoch遍历，中的每一个sample的遍历 都会改变权重，但是每一次perdict的时候，只有所有的update更新之后才计算预测值



