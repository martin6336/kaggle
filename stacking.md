# stacking
stacking方法关键思想就是利用前一层模型的结果作为后面层的feature。二分类问题中可以直接拿预测出来是0还是1当新的feature，多分类问题讲道理也可以这么搞。
不要感觉怪，又不是去回归得出1，2,3，直观来讲就是某个样本的某个feature是0，1，2，3那他是对应类别的概率当然很大，不要被你那种回归的想法制约。**分类器预测的不是1，2，3，他预测出来的是属于各个类别的概率**
注意stacking其实就是给模型换一些浓缩的feature，所以这个feature不是一定只有一个（类别），也可以是多列像各个类别的概率。并且也可以添加其他天然的feature，这个就仁者见仁了。
# LR应用到多分类问题
http://deeplearning.stanford.edu/wiki/index.php/Softmax_Regression

目标函数和算各个概率的函数都要改变，这时候就是一个两层的nn

# SGD
http://www.datakit.cn/blog/2016/07/04/sgd_01.html

引入momentum
![](http://www.datakit.cn/images/machinelearning/sgd_momentum_2.png)

Nesterov accelerated gradient，对SGD进一步改进
# 朴素贝叶斯
[阮一峰](http://www.ruanyifeng.com/blog/2013/12/naive_bayes_classifier.html)

# factorization machine
[peghoty](http://www.cnblogs.com/pinard/p/6370127.html)
[美团](https://tech.meituan.com/deep-understanding-of-ffm-principles-and-practices.html#mjx-eqn-eqfm)

这个方法其实就是对于引入交叉项的回归方程，当他要去处理很稀疏的矩阵时，交叉项很多都是0，这样他是没法估计交叉项系数的。对于每一列的特征，引入一个相同维度的向量，系数就是对应两个向量相乘。这样既节省了计算，又能估计这些稀疏的系数。

其原因是，每个参数 wij 的训练需要大量 xi 和 xj 都非零的样本；由于样本数据本来就比较稀疏，满足“xi 和 xj 都非零”的样本将会非常少。训练样本的不足，很容易导致参数 wij 不准确，最终将严重影响模型的性能。

**总之首先是有想引入交叉项的想法，毕竟影响因变量可能交叉项的效果更大，这个想法不管是回归还是分类都是自然的。但是对于稀疏矩阵有一个问题就是交叉项很多都是0，这是没法子训练系数的，fm就是为了解决这个问题，也就是换种方法计算各个交叉项的参数，这个想法当然同时适用于回归以及分类问题。**
