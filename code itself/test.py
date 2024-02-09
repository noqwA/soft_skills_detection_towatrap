import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# загрузка данных тестирования из CSV-файла
data_for_testing = pd.read_csv('pythonProject/porttfolio/db_for_testing_psy_1.csv')

# предварительная обработка данных
test_inputs = data_for_testing['Input']
test_labels = data_for_testing['Diagnosis']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(test_inputs)
test_sequences = tokenizer.texts_to_sequences(test_inputs)
data_for_testing = pad_sequences(test_sequences, maxlen=100)

# загрузка модели
model = tf.keras.models.load_model('psy_recog_2.krs')

# тестирование модели
predictions = model.predict(data_for_testing)

# вывод результатов
labels = [np.argmax(i)for i in predictions]

for i, prediction in enumerate(labels):
    print("Data Input: {}".format(test_inputs[i]))
    print("Prediction: {}".format(prediction))
    print("Actual Diagnosis: {}\n".format(test_labels[i]))

