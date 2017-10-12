GitHub上有一篇12th的code，看了之后发现他没用什么高大上的方法，就两个xgboost，nn，
但是有几个亮点：
1，他的feature engineer也很笼统，就是构造出一个新的矩阵和原矩阵相乘得到新的数据集，作用就是各个原特征相互相加减。但从结果来说效果很好。
2，他用的pylearn，能更详细的操作nn的内部。
3，他自己编程实现找到最优参数，没用很多模型但是很多精力给了调参。
4，最后就是xgboost和nn加权各个概率


还有一位66位的大神，![传送门](http://blog.aicry.com/kaggle-otto-group-product-classification-challenge/)。calibration这个库很有意思。 注意的点：
1，他的stacking预测数据也是保证train过的数据不会用他拟合的模型去预测，对于train的运算和你一样，但是test数据集他是直接用所有trian训练然后预测，不是像你那样使用kfold中的各个train子集预测后在平均。
