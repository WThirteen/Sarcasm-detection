from keras.models import load_model  
from keras.preprocessing.text import tokenizer_from_json  
import numpy as np  
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  
import json  

# 从文件读取 JSON 字符串  
with open('tokenizer_config.json', 'r', encoding='utf-8') as f:  
    # tokenizer_config = json.load(f)  
    tokenizer_config = f.read()

# 使用配置创建新的Tokenizer对象  
loaded_tokenizer = tokenizer_from_json(tokenizer_config)  


# 加载整个模型（包括结构和权重）  
model = load_model('my_model.h5')  

new_texts = ['nice to meet you']  

# new_sequences = tokenizer.texts_to_sequences(new_texts)  
new_sequences = loaded_tokenizer.texts_to_sequences(new_texts)
# max_len = max([len(seq) for seq in new_sequences])
new_padded_sequences = pad_sequences(new_sequences, maxlen=176)

# 进行预测  
predictions = model.predict(new_padded_sequences)  
  
predicted_classes = np.argmax(predictions, axis=1)  

# 将这些整数转换回原始的标签名  
label_to_name = {0: '非讽刺', 1: '讽刺'}  
predicted_labels = [label_to_name[cls] for cls in predicted_classes]

print(predicted_labels)
