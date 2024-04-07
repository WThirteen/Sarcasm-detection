数据集：
Sarcasm Corpus V2


使用该数据集进行训练
得到模型，并对其进行检测。

训练模型的准确率和损失：

![屏幕截图 2024-04-07 164841](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/8749563d-a20e-41aa-a0c5-f33e9333708d)


数据集和已经训练好的模型放在百度网盘：

链接: https://pan.baidu.com/s/1Btkmx-3orPr5zLrbz-k9qA 提取码: xnyq 


python版本 3.10

numpy 1.24.4

keras 2.10.0

tensorflow-gpu 2.10.0


code1是原始版本，在使用已经训练好的模型和tokenizer时出现问题。

code1_v2已经修改了这个问题。


训练模型可直接使用命令

python train.py

需下载数据集，将data与train.py放在同一文件夹下。
