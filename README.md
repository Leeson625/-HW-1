# 从零开始构建三层神经网络分类器，实现图像分类
> 模块实现部分参考书籍《深度学习入门：基于Python的理论与实现 (斋藤康毅)》

### common文件夹中为各模块文件：
- functions 为必要的函数模块，并实现了交叉熵损失函数和softmax函数
- gradient 为数值微分求导模块，利用微小的差分求导数
- layers 实现了必要的神经网络的层类，如Relu、Sigmoid、Affine 和SoftmaxWithLoss。其中每个函数均通过forword方法实现前向计算，backward方法实现反向传播来传递导数。在模块中也定义了Dropout，BatchNormalization，卷积层和池化层
- util 模块中定义了打乱数据集的函数，im2col实现卷积层展开快速矩阵运算的函数，col2im将运算后结果再次重构回原始结构
- optimizer 模块定义各优化方法的类，如SGD、Momentum、AdaGrad、Adam等，被trainer类调用以实现模型的训练过程
- multi_layer_net 模块实现了全连接的多层神经网络，具体参数在函数中有说明
- multi_layer_net_extend 实现了扩展版的全连接的多层神经网络，具有Weiht Decay、Dropout、Batch Normalization的功能，具体参数在函数中有说明
- trainer 定义了进行神经网络训练的类，可以根据验证集指标自动保存最优的模型权重

### Fashion_MNIST为数据集文件：
- 里面有原始数据和传入数据的函数文件 <br> 
- 函数参数：<br> 
   normalize : 将图像的像素值正规化为0.0~1.0  <br> 
   one_hot_label : one_hot_label为True的情况下，标签作为one-hot数组返回  <br> 
   flatten : 是否将图像展开为一维数组  <br>

### 训练和测试：
- 通过multi_layer_net导入MultiLayerNet类，进行神经网络的定义，如隐藏层大小、激活函数类型，输出大小，和参数衰减权重
- 然后通过trainer模块中Trainer类，进行神经网络的训练，并根据验证集指标自动保存最优的模型权重
- 神经网络类具有保留参数和加载参数的方法，可以将模型参数加载并在测试集上进行推理
- 详细过程实现可参考HW-1文件
