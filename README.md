# 从零开始构建三层神经网络分类器，实现图像分类

### 其中common文件夹中为各模块：
- functions 为必要的函数模块
- gradient 为数值求导模块
- layers实现了必要的神经网络的层类，如Relu、Sigmoid、Affine 和SoftmaxWithLoss
- util实现了打乱数据集的函数
- optimizer为各优化方法的类，如SGD、Momentum、Adam，被trainer类所调用
- multi_layer_net实现了全连接的多层神经网络，具体参数在函数中有说明
- multi_layer_net_extend实现了扩展版的全连接的多层神经网络，具有Weiht Decay、Dropout、Batch Normalization的功能，具体参数在函数中有说明
- trainer定义了进行神经网络训练的类

### 数据集：
- 首先定义
