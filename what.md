GitHub上有一篇12th的code，看了之后发现他没用什么高大上的方法，就两个xgboost，nn，
但是有几个亮点：

1，他的feature engineer也很笼统，就是构造出一个新的矩阵和原矩阵相乘得到新的数据集，作用就是各个原特征相互相加减。但从结果来说效果很好。

2，他用的pylearn，能更详细的操作nn的内部。

3，他自己编程实现找到最优参数，没用很多模型但是很多精力给了调参。

4，最后就是xgboost和nn加权各个概率


还有一位66位的大神，![传送门](http://blog.aicry.com/kaggle-otto-group-product-classification-challenge/)。calibration这个库很有意思。 注意的点：

1，他的stacking预测数据也是保证train过的数据不会用他拟合的模型去预测，对于train的运算和你一样，但是test数据集他是直接用所有trian训练然后预测，不是像你那样使用kfold中的各个train子集预测后在平均。

2，另外就是 OneVsRestClassifier，有两个好处总共就需要拟合n_class个模型（computational efficiency），另外有更好的解释性。这个几乎是为svm定制的，svm最初就是解决二分类问题，多分类计算复杂度太高，one-vs-rest就可以解决这个问题。

3，使用calibration

4,作者自己编了个rgf函数，分类用的，Regularized Greedy Forest以后看

5，编程结构把基础模型分别放在不同的路径下，虽然可能很散，并且看起来代码很多，但是很系统对于每个基础模型都可以进行详细的调试，并且也可以分别输出各个模型表现，实际很方便。不同基模型的代码结构也类似。这样也更容易对每个基模型进行调参实现更好的结果，也容易增加减少模型，或者对模型进行更深入的操作，或者自己编写模型

6，让人很在意的是大神们好像都特喜欢自己编出来个模型，不管是nn还是xgboost

7，LinearSVC.fit_transform特征选择

*todo* 1，calibration 2，svm（one-vs-rest) 3,RI矩阵构造新数据阵 4，test数据集的kfold处理
