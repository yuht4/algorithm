import numpy as np

class AdalineGD():

	#### rate 是学习率 number是epoch迭代的次数
	def __init__(self, rate = 0.01, number_iter = 50):
		self.rate = rate;
		self.number_iter = number_iter;
	
	#### 首先建立一个weight_列表，长度为X矩阵的第二个维度的长度 + 1， 要多一个阈值，首先全部设置为0
	#### cost_ 列表 表示的是每一次epoch对应的cost函数的大小

	#### 注意之前的perceptron算法是通过每一次的sampla来upate权重
	#### 这里是通过所有的data set来更新weight
	def fit(self, X, y):

		self.wight_ = np.zeros(1 + X.shape[1]);
		self.cost_ = [];

		for i in range(self.number_iter):

			output = self.net_input(X); #### output是一个一维矩阵 vector，对应不同sample的预测值
			errors = (y - output); #### 这个是对应的误差矩阵，注意这里使用 +1 -1 与 net_input值进行相减，为了让net_input更接近1 -1

            
            #### array 的 T 函数相当于是转置 将同一列的作为一行拿出来
            #### 这样相当于得到了同一features下的所有行 不同的值，这些值是根据sample number 排列的

			self.wight_[1:] += self.rate * X.T.dot(errors); ### 这里是features的多维矩阵 和 errors的一维矩阵dot，得到的是不同features下的 update值 的矩阵
			#### 然后矩阵之间相互加减

			self.wight_[0] += self.rate * errors.sum();
			#### 因为阈值 对应的 x0 是 1 ， 所以之间求和就行了

			cost = (errors ** 2).sum() / 2.0;  ### cost calculate
			self.cost_.append(cost);

		return self;


	#### 这里是多维矩阵和 一维矩阵的dot
	#### 最后得出的是一个一维矩阵，对应不同的sample number之下的 WX net_input 的值
	#### array([1,2,3,4,5])
	#### 注意加上 阈值 wight_[0]
	def net_input(self, X):
		return np.dot(X, self.wight_[1:]) + self.wight_[0];
    


    #### 这里注意activation function 和 net_input 相同
	def predict(self, X):
		return np.where(self.net_input(X) >= 0.0 , 1, -1);