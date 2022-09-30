# Tensorflow Tutorial (NLP Zero to Hero - Part 6)
# Training an AI to create poetry

import tensorflow as tf 
from keras_preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

tokenizer = Tokenizer()

data = open('Harsha_Quotes.txt').read()
corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus) 
total_words = len(tokenizer.word_index) + 1 

# Setup Training Data
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1] 
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding = 'pre'))

xs = input_sequences[:, :-1]
labels = input_sequences[:, -1] 
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = Sequential()
model.add(Embedding(total_words, 240, input_length = max_sequence_len-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_words, activation = 'softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)

def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.show() 

plot_graphs(history, 'accuracy')
# Test Data 
seed_text = "Sachin Tendulkar and Rahul Dravid were"
next_words = 100

for _ in range(next_words): 
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen = max_sequence_len-1, padding = 'pre')
    predicted = np.argmax(model.predict(token_list), axis = -1)
    output_word = " "
    for word, index in tokenizer.word_index.items():
        if index == predicted: 
            output_word = word 
            break 
    seed_text += " " + output_word 
print(seed_text)


#Results

""" Sachin Tendulkar and Rahul Dravid were you add to the moment in the crowd did in the team cricketer ever in the world than could have won the same talent the man sitting next to me for tomorrow's newspaper to declare him out out 50 000 people in the crowd did to tendulkar in difficult times don't build character it reveals 4th gear than the ball man sitting next to me just on the map otherwise it is just india and pakistan pakistan pakistan pakistan the crowd did in numbers the ball in the won than the won the won man won in the world he """