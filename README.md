# 1.数据集：
* Sarcasm Corpus V2
使用该数据集进行训练
得到模型，并对其进行检测。

# 2.训练模型的准确率和损失：

![training_1](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/8749563d-a20e-41aa-a0c5-f33e9333708d)  
## 使用train_v3.py训练模型的准确率和损失：
*修正了数据集的标签
![training_2](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/7f30f7bd-62b3-48ae-ab46-acae7e5cadaa)

# 3.数据集和已经训练好的模型放在百度网盘：

链接: https://pan.baidu.com/s/1Btkmx-3orPr5zLrbz-k9qA 提取码: xnyq  
原模型的名字为 _my_model_old.h5_  
用train_v3.py训练的模型为 _my_model.h5_
# 4.版本
python版本 3.10  
numpy 1.24.4  
keras 2.10.0  
tensorflow-gpu 2.10.0

# 更新
code1是原始版本，在使用已经训练好的模型和tokenizer时出现问题。

code1_v2已经修改了这个问题。  
在predicate_v2.py中 更新 可输入 自定义语句 来判断是否讽刺。  
## train_v3.py中,修改了数据集标签的问题。
* train_v2.py 与 train.py 的数据集的标签都存在问题  
* 仅train_v3.py修改了这个问题  


# 使用方法
## 训练模型
可直接使用命令  
`python train_v3.py`  
_需下载数据集，将data与train.py放在同一文件夹下。_

## 讽刺判断 
* 1.predicate.py 版本需修改代码中的 *new_texts*   
* 2.predicate_v2.py 版本 在运行后 通过输入自定义语句来判断是否讽刺  
如想结束输入 使用*exit*结束输入，并对已输入的数据进行预测。
