# Sarcasm-detection
<h4 align="center">
    <p>
        <b>中文</b> |
        <a href="(https://github.com/WThirteen/Sarcasm-detection/edit/main/README_EN.md">English</a>
    <p>
</h4>

</div>   

# 讽刺检测  
## 1.数据集：
* Sarcasm Corpus V2  
使用该数据集进行训练
得到模型，并对其进行检测。
* 若想查看讽刺语句 可运行view_data  
依次运行每一块即可

## 2.训练模型的准确率和损失：

![training_1](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/8749563d-a20e-41aa-a0c5-f33e9333708d)  
### 使用train_v3.py训练模型的准确率和损失：
* 修正了数据集的标签   
![training_2](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/7f30f7bd-62b3-48ae-ab46-acae7e5cadaa)  
### 使用train_v5.py训练模型：
* 第一次训练
![train_v5_v1](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/c036b49a-7e24-4b83-b88a-ca4b79cafc6e)
* 第二次训练
![train_v5_v2](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/b16a3e67-f96f-49a7-ac7e-81aa7f5752e4)  
* 第三次训练
![train_v5_v3](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/29a67b46-3f0e-46ce-bca0-5e50fb69ff6b)  

## 3.数据集和已经训练好的模型放在百度网盘：

链接: https://pan.baidu.com/s/1Btkmx-3orPr5zLrbz-k9qA 提取码: xnyq  
原模型的名字为 _my_model_old.h5_  
用train_v3.py训练的模型为 _my_model.h5_
#### 后续更新训练的模型仍会放置在百度网盘（模型名字会使用训练的train版本来命名）
train_v5.py第一次训练 即 my_model_v5_v1.h5

## 4.版本
python版本 3.10  
numpy 1.24.4  
keras 2.10.0  
tensorflow-gpu 2.10.0  
### 使用命令配置环境  
```
pip install -r requirements.txt
```


## 更新
code1是原始版本，在使用已经训练好的模型和tokenizer时出现问题。

code1_v2已经修改了这个问题。  
在predicate_v2.py中 更新 可输入 自定义语句 来判断是否讽刺。  
### train_v3.py中,修改了数据集标签的问题。
* train_v2.py 与 train.py 的数据集的标签都存在问题  
* train_v3.py修改了这个问题
### train_v4.py 在v3的基础上，增加了config，可通过修改config中参数修改训练轮次、数据集位置  
### train_v5.py 在v4的基础上，修改了模型，增加了回调函数。

# 使用方法
## 训练模型
可直接使用命令  
```
python train_v3.py
```
_需下载数据集，将data与train.py放在同一文件夹下。_  

或直接使用  
```
python train_v4.py
```
_可在config中修改数据集位置_  

## 讽刺判断 
* 1.predicate.py 版本需修改代码中的 *new_texts*   
* 2.predicate_v2.py 版本 在运行后 通过输入自定义语句来判断是否讽刺  
如想结束输入 使用*exit*结束输入，并对已输入的数据进行预测。
