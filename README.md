# 从零开始构建三层神经网络分类器，实现图像分类

### common文件夹中为各模块文件夹：
- functions 为必要的函数模块
- gradient 为数值求导模块
- layers实现了必要的神经网络的层类，如Relu、Sigmoid、Affine 和SoftmaxWithLoss
- util实现了打乱数据集的函数
- optimizer为各优化方法的类，如SGD、Momentum、Adam，被trainer类所调用
- multi_layer_net实现了全连接的多层神经网络，具体参数在函数中有说明
- multi_layer_net_extend实现了扩展版的全连接的多层神经网络，具有Weiht Decay、Dropout、Batch Normalization的功能，具体参数在函数中有说明
- trainer定义了进行神经网络训练的类

### Fashion_MNIST为数据集文件：
- 里面有原始数据和传入数据的函数文件 <br> 
- 函数参数：<br> 
   normalize : 将图像的像素值正规化为0.0~1.0  <br> 
    one_hot_label : one_hot_label为True的情况下，标签作为one-hot数组返回  <br> 
    flatten : 是否将图像展开为一维数组  <br>

### 训练和测试：
- 通过multi_layer_net导入MultiLayerNet类，进行神经网络的定义
- 然后通过trainer模块中Trainer类，进行神经网络的训练
- 神经网络类具有保留参数和加载参数的方法
- 详细过程可参考HW-1文件
