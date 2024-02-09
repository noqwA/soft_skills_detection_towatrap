import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

# 1. загрузка датасета для тестирования
dataset = pd.read_csv('pythonProject\porttfolio\db_full_psy_v2.csv') 

# 2. предварительная обработка данных:

inputs = dataset['Input']
labels = dataset['Diagnosis']


print(labels[0])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(inputs)
sequences = tokenizer.texts_to_sequences(inputs)
data = pad_sequences(sequences, maxlen=100)

labels = to_categorical(labels)


print(len(tokenizer.word_index))
# 3. создание модели нейронной сети:

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+2, input_length=100, output_dim=16))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)))
model.add(Dense(16, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# 4. обучение модели и ее сохранение

model.fit(data, labels, epochs=25, batch_size=8)
model.save('psy_recog_2.krs')

# api external servises (Sber - voice recognition(100min_free))
# Libraries: speech recognition    pip install SpeechRecognition      scikit learn train test split