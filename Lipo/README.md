<center><h1>GEM</h1></center>

这是百度 [Geometry-enhanced molecular representation learning for property prediction](https://www.nature.com/articles/s42256-021-00438-4) 论文代码的整理版本，原始代码在 [Paddlehelix](https://github.com/PaddlePaddle/PaddleHelix) 。

因为该仓库包含子仓库（PaddleHelix），因此 clone 的时候应该使用递归命令来同时将子仓库 clone 下来：

```
git clone --recurse-submodules git@github.com:fanxiaoyu0/GEM.git
```

### 示例项目

Lipo 文件夹下是使用 GEM 预测小分子的亲脂性（Lipophilicity）的一个例子，其中 lipo 数据集来自 MoleculeNet（包含多个小分子性质预测的 benchmark 数据集的平台，广泛被学界使用）。下面介绍一下各个文件的含义：

```
/data/    存放数据集相关文件
/data/raw/    存放原始数据文件，例如从 MoleculeNet 上下载下来的 .csv 文件
/data/intermediate/    存放运行模型所需要的中间文件，通常为 .pkl 文件  
/data/result/    存放结果文件，例如对实验结果的可视化

/lib/    存放运行模型需要的库文件，通常为无法通过 pip install 直接安装的 github 项目，一般不需要修改
/lib/PaddleHelix/    PaddleHelix 的 github 仓库，其中包含我们需要调用的函数，PaddleHelix 发布到 pypi 上的版本还不支持 GEM

/src/    存放代码文件
/src/main.py    项目主代码文件，因为所有的数据集都已经准备好，python main.py 即可运行项目

/weight/    存放模型权重
/weight/PaddleHelix/    存放 GEM 的预训练权重，在 PaddleHelix 仓库中有预训练权重的下载地址

clean.sh    清除所有数据文件和模型权重文件，以便你将其替换为自己的数据
.gitmodule    因为该仓库包含子仓库（PaddleHelix），所以需要该文件声明子仓库信息。
README.md    介绍项目概况
```

运行 Lipo 项目的方式：

```
python main.py
```

### 运行自己的项目

一种可能的使用方法：

将 Lipo 项目的内容复制一份，作为自己的项目文件夹，然后在自己的项目文件夹中进行修改，例如更换数据集、改写代码，这样的好处是修改之后如果出错可以与 Lipo 项目进行对照。所需要的命令大致如下：

```
# 复制项目文件
cp -r Lipo xxxx(your project name)
# 删除 Lipo 文件夹的数据集和模型权重文件，以便更换成自己的数据集
cd xxxx
bash clean.sh
```
