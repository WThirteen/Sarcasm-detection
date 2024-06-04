# Sarcasm-detection
<h4 align="center">
<p>
<a href="(https://github.com/WThirteen/Sarcasm-detection/edit/main/README.md">中文</a> |
<b> English</b>
<p>
</h4>

</div>

# Irony detection
## 1. Data set:
* Sarcasm Corpus V2
Use this data set for training
The model is obtained and tested.
* To view ironic statements, run view_data
Just run each piece in turn

## 2. Accuracy and loss of training model:

! [training_1](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/8749563d-a20e-41aa-a0c5-f33e9333708d)
### Accuracy and loss of training model using train_v3.py:
* Fixed the label of the data set
! [training_2](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/7f30f7bd-62b3-48ae-ab46-acae7e5cadaa)
### Use train_v5.py to train the model:
* First training session
! [train_v5_v1](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/c036b49a-7e24-4b83-b88a-ca4b79cafc6e)
* Second training
! [train_v5_v2](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/b16a3e67-f96f-49a7-ac7e-81aa7f5752e4)
* Third training session
! [train_v5_v3](https://github.com/WThirteen/Sarcasm-detection/assets/100677199/29a67b46-3f0e-46ce-bca0-5e50fb69ff6b)

## 3. Put the data set and the trained model on Baidu web disk:

Link: https://pan.baidu.com/s/1Btkmx-3orPr5zLrbz-k9qA extraction code: xnyq
The name of the original model is _my_model_old.h5_
The model trained with train_v3.py is _my_model.h5_
#### The updated and trained model will still be placed on the Baidu web disk (the model name will be named with the train version of the training).
train_v5.py First training is my_model_v5_v1.h5

## 4. Version
python version 3.10
numpy 1.24.4
keras 2.10.0
tensorflow-gpu 2.10.0
### Configure the environment using commands
` ` `
pip install -r requirements.txt
` ` `


## Update
code1 is the original version and has problems using models and Tokenizers that have already been trained.

code1_v2 has fixed this issue.
Updates in predicate_v2.py allow you to enter custom statements to determine sarcasm.
In ### train_v3.py, fixed an issue with data set labeling.
* Both train_v2.py and train.py data sets have problems with labeling
* train_v3.py fixed this problem
train_v4.py added config on the basis of v3. You can modify the training rounds and data set location by modifying the parameters in config
train_v5.py modified the model and added callback functions on the basis of v4.

# How to use
## Train the model
Commands can be used directly
` ` `
python train_v3.py
` ` `
_ To download the data set, place the data in the same folder as train.py. _

Or direct use
` ` `
python train_v4.py
` ` `
_ Data set location can be modified in config _

## Sarcastic judgment
* 1.predict.py version requires changes to *new_texts* in the code
* 2.predicate_v2.py version determines sarcasm by typing custom statements after running
If you want to end the input, use *exit* to end the input and make a prediction of the entered data.
