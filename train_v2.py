import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer   
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential  
from keras.layers import Embedding, LSTM, Dense  
from sklearn.model_selection import train_test_split  
import json  
from keras.models import load_model 


def load_data():
    
    # 导入数据 处理数据
    d_gen=pd.read_csv('data/GEN-sarc-notsarc.csv')
    d_hyp=pd.read_csv('data/HYP-sarc-notsarc.csv')
    d_rq=pd.read_csv('data/RQ-sarc-notsarc.csv')

    text1=d_gen["text"]
    text2=d_hyp["text"]
    text3=d_rq["text"]

    # 设置1为讽刺
    # 0为非讽刺
    lable1=[1]*len(text1)
    lable2=[0]*len(text2)
    lable3=[0]*len(text3)

    # 合并
    texts=[]
    labels=[]
    texts.extend(text1)
    texts.extend(text2)
    texts.extend(text3)
    labels.extend(lable1)
    labels.extend(lable2)
    labels.extend(lable3)

    return texts,labels

def creat_token(texts):
    # 使用Keras的Tokenizer对句子进行编码
    # 初始化 Tokenizer 并拟合文本数据  
    tokenizer = Tokenizer(num_words=10000)  # 假设我们只想保留最常见的10000个单词  
    tokenizer.fit_on_texts(texts)  

    # 将 Tokenizer 配置转换为 JSON 字符串  
    tokenizer_config = tokenizer.to_json() 

    # 将 JSON 字符串保存到文件  
    with open('tokenizer_config.json', 'w', encoding='utf-8') as f:  
        f.write(tokenizer_config) 


    # 将文本转换为序列  
    sequences = tokenizer.texts_to_sequences(texts)  
  
    # 填充/截断序列，使其具有相同的长度  
    max_len = max([len(seq) for seq in sequences])  
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')  

    return tokenizer,sequences,max_len,padded_sequences

def prepare_data(padded_sequences,labels):
    # 假设 padded_sequences 是经过填充的序列，labels 是对应的标签  
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test  


def model_train(tokenizer,max_len,X_train, X_test, y_train, y_test):
    # 定义模型  
    model = Sequential()  
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len))  
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  
    model.add(Dense(1, activation='sigmoid'))  # 假设是二分类问题  
  
    # 编译模型  
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  

    y_test = np.array(y_test)
    y_train = np.array(y_train)

    # 训练模型  
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)  

    # 保存模型
    model.save('my_model.h5')

def model_eval(X_test, y_test):
    model = load_model('my_model.h5')  
    # 评估模型  
    _, accuracy = model.evaluate(X_test, np.array(y_test))  
    print(f'Test accuracy: {accuracy:.4f}')

def run():
    print("---load_data---")
    texts,lables = load_data()
    print("---creat_token---")
    tokenizer,sequences,max_len,padded_sequences = creat_token(texts)
    print("---prepare_data---")
    X_train, X_test, y_train, y_test = prepare_data(padded_sequences,lables)
    print("---train---")
    # model_train(tokenizer,max_len,X_train, X_test, y_train, y_test)
    print("---model_eval---")
    model_eval(X_test, y_test)



if __name__ == "__main__":
    run()
    
    