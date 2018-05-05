#### 感知器算法  perceptron learning

features [x1 , x2, x3, ..., xi]
weights  [w1, w2, w3, ...., wi]


***********
F(z) 用于判断是否成立 activition function

z = x1 * w1 + x2 * w2 + x3 * w3 + .... + xi * wi

F(z) = 1 if z >= threshold or -1


***********
主要步骤

1, 将所有的weight w1, w2, ..., wi 分布为0
2, 对每一个样本