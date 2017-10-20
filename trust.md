对于AUC指标来讲，最好的当然是正的指标越接近1，但是你把预测出来的大于某个阈值的都往上调会出问题的，因为他可能并不属于这个类，这时候的惩罚是很大的。你对了，
当然你的概率越大越好，就怕你是错的，你还特别自信，这时候惩罚很大。所以对于auc来讲，个人觉着最真实的概率就好，不是很把握那我预测的概率就小一点。

TypeError: 'generator' object is not subscriptable,hyperopt error because networkx version
sudo pip uninstall networkx
sudo pip install networkx==1.11
