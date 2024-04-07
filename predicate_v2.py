from keras.models import load_model  
from keras.preprocessing.text import tokenizer_from_json  
import numpy as np  
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import json  

def read_token():

    # 从文件读取 JSON 字符串  
    with open('tokenizer_config.json', 'r', encoding='utf-8') as f:  
        # tokenizer_config = json.load(f)  
        tokenizer_config = f.read()

    # 使用配置创建新的Tokenizer对象  
    loaded_tokenizer = tokenizer_from_json(tokenizer_config)  

    return loaded_tokenizer
   
def IN():  
    new_texts = []  
    print("--输入'exit'退出--")
    while True:  
        print("请输入句子：")  
        temp = input()  
        if temp.lower() == "exit":   
            break  
        new_texts.append(temp)  # 直接将输入的句子作为字符串添加到列表中  
    return new_texts 


def pred(loaded_tokenizer , new_texts):
    # 加载整个模型（包括结构和权重）  
    model = load_model('my_model.h5')  

    new_sequences = loaded_tokenizer.texts_to_sequences(new_texts)

    new_padded_sequences = pad_sequences(new_sequences, maxlen=176)

    # 进行预测  
    predictions = model.predict(new_padded_sequences)  
  
    predicted_classes = np.argmax(predictions, axis=1)  

    # 将这些整数转换回原始的标签名  
    label_to_name = {0: '非讽刺', 1: '讽刺'}  
    predicted_labels = [label_to_name[cls] for cls in predicted_classes]
    print(predicted_labels)


def run():
    new_texts = IN()
    loaded_tokenizer = read_token()
    pred(loaded_tokenizer , new_texts)


if __name__ == "__main__":
    run()

