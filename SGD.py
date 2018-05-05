from numpy.random import seed

class AdalineSGD():

	def __init__(self, eta = 0.01, n_iter = 10, shuffle = True, random_state = None):
		
		self.eta = eta;
		self.n_iter = n_iter;
		self.shuffle = shuffle;
		self.random_state = random_state; ### 作为随机种子 用于随机化处理

	    if random_state:
	    	seed(random_state);

	def fit(self, X, y):

		self._initialize_weights(X.shape[1]);
		
        #### 存放每一次epoch对应的cost值，注意这里的cost是指所有的sample的平均值，因为这是随机的，每一个sample都要update一次
		self.cost_ = [];

		for i in range(self.n_iter):
			if self.shuffle:
				X, y = self._shuffle(X, y);

			### 存放每一次epoch过程中 的每一个sample对应的cost
			cost = [];

			for xi, target in zip(X, y):
				cost.append(self._update_weights(x1, target));

			#### 求出一个epoch的平均cost
			avg_cost = sum(cost) / len(y);

			#### 添加epoch对应的平均cost
			self.cost_.append(avg_cost);
		return self;
    

    ###3 针对online update，这是一个多的sample 针对单个sample来说
    #### 这个方法不会重新初始化weights 而只是使用模型的weights
	def partial_fit(self, X, y):
		if not self.w_initalized:
			self._initialize_weights(X.shape(1));
		if y.ravel().shape[0] > 1:
			for x1, target in zip(X, y):
				self._update_weights(x1, target);
		else:
			self._update_weights(X, y);
		return self;

    #### 初始化 Weights list
    def _initialize_weights(self, m):
    	self.w_ = np.zeros(1 + m);
    	self.w_initalized = True;

    def _update_weights(self, x1, target):
    	#### 对应单个sample update weights
    	output = self.net_input(x1);
    	error = target - output; ### 得到target 和 netinput的差别

    	self.w_[1:] += self.eta * x1.dot(error); ### 针对每一个权重 计算 update 然后加上
    	self.w_[0] += self.eta * error; ### 对于阈值，由于对应的x0 是1 所以直接乘法就可以了

    	cost = 0.5 * error ** 2; #### 计算这个sample之下的cost

    	return cost;


  
    #### 用于随机打乱X 和 y 的矩阵顺序
    def _shuffle(self, X, y):

    	#### 产生一个算计的random sequence
    	r = np.random.permutation(len(y));
    	return X[r], y[r];
    	#### 然后使用 r这个随机序列 作为指数来suffle 我们的矩阵

    	#### 这里是为了保持同样的顺序，按照同样的模式进行打乱


    #### 获得net inpupt的结果 W X, 注意加上阈值
    def net_input(self, X):
    	return np.dot(X, self.w_[1:]) + self.w_[0];

    #### 激活函数，和 net input 函数相同
    def activation(self, X):
    	return self.net_input(X);

    #### 判断的标准就是net_input 是否大于0， 因为已经加上了阈值的计算， 得到的 class label
    def predict(self, X):
    	return np.where(self.activation(X) >= 0.0, 1, -1);

