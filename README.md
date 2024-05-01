# 从零开始构建三层神经网络分类器，实现图像分类

#### 其中common文件夹中为各模块：
- functions 为必要的函数模块
- gradient 为数值求导模块
- layers实现了必要的神经网络的层类，如Relu、Sigmoid、Affine 和SoftmaxWithLoss
- util实现了打乱数据集的函数
- optimizer为各优化方法的类，如SGD、Momentum、Adam，被trainer类所调用
- multi_layer_net实现了全连接的多层神经网络，各参数分为：
   Parameters
   input_size : 输入大小（MNIST的情况下为784）
    hidden_size_list : 隐藏层的神经元数量的列表（e.g. [100, 100, 100]）
    output_size : 输出大小（MNIST的情况下为10）
    activation : 'relu' or 'sigmoid'
    weight_init_std : 指定权重的标准差（e.g. 0.01）
        指定'relu'或'he'的情况下设定“He的初始值”
        指定'sigmoid'或'xavier'的情况下设定“Xavier的初始值”
    weight_decay_lambda : Weight Decay（L2范数）的强度
- 
