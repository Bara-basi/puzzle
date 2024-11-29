# 配置需求
- Python 3.7+
- cuda 12.5+
- pytorch 2.3+
- tqdm 4.62.3+
- opencv-python 4.10.0.84+
- numpy 1.26.4+
- pillow 10.4.0+
- matplotlib 3.9.1+
- 显存需求：4G+, 显存不足可减少BATCH_SIZE（config.py中设置）

# 运行方式
- 模型已训练并保存，可直接运行`main.py`进行测试。
- 训练模型：删除`models`文件夹下的`final_model.pth`,重新运行`main.py`，选择`models`文件夹下生成的模型文件`.pth`,重命名为`final_model.pth`,运行`main.py`即可。 
# 模块简介
使用机器学习经典的pipeline写法
- config.py: 配置文件，设置训练参数
- loader.py: 数据处理模块，包括数据加载、数据增强、数据集划分、数据集构建
- model.py: 模型定义模块，包括模型结构、损失函数、优化器
- main.py: 主程序，包括训练、测试、模型保存、模型加载等功能
- evaluate.py: 评估模块，包括模型评估、结果可视化等功能
- agent.py: 智能体模块，包含智能体和智能体环境
# 数据集
- 来源：https://modelscope.cn/datasets/xsyl06/flower_data
- 位置：img文件夹中，train.csv、val.csv保留图片的路径
- 格式：jpg图片