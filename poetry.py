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

""" data="Come all ye maidens young and fair \n And you that are blooming in your prime \n Always beware and keep your garden fair \n Let no man steal away your thyme \n For thyme it is a precious thing \n And thyme brings all things to my mind \n nlyme with all its flavours, along with all its joys \n Thyme, brings all things to my mind \n Once I and a bunch of thyme \n i thought it never would decay \n Then came a lusty sailor \n Who chanced to pass my way \n And stole my bunch of thyme away \n The sailor gave to me a rose \n A rose that never would decay \n He gave it to me to keep me reminded \n Of when he stole my thyme away \n Sleep, my child, and peace attend thee \n All through the night Guardian angels God will send thee \n Soft the drowsy hours are creeping \n Hill and dale in slumber sleeping \n I my loving vigil keeping \n While the moon her watch is keeping \n While the weary world is sleeping \n Oer thy spirit gently stealing \n Visions of delight revealing \n Breathes a pure and holy feeling \n Though I roam a minstrel lonely \n My true harp shall praise sing only \n Loves young dream, alas, is over \n Yet my strains of love shall hover \n Near the presence of my lover \n Hark, a solemn bell is ringing \n Clear through the night \n Thou, my love, art heavenward winging \n Home through the night \n Earthly dust from off thee shaken \n Soul immortal shalt thou awaken \n  With thy last dim journey taken \n  Oh please neer forget me though waves now lie oer me \n  I was once young and pretty and my spirit ran free \n  But destiny tore me from country and loved ones \n  And from the new land I was never to see. \n  A poor emigrants daughter too frightened to know \n  I was leaving forever the land of my soul \n  Amid struggle and fear my parents did pray \n  To place courage to leave oer the longing to stay \n They spoke of a new land far away cross the sea \n And of peace and good fortune for my brothers and me \n So we parted from townland with much weeping and pain \n Kissed the loved ones and the friends we would neer see again. \n The vessel was crowded with disquieted folk \n The escape from past hardship sustaining their hope \n But as the last glimpse of Ireland faded into the mist \n Each one fought back tears and felt strangely alone. \n The seas roared in anger, making desperate our plight \n And a fever came oer me that worsened next night \n Then delirium possessed me and clouded my mind \n And I for a moment saw that land left behind. \n I could hear in the distance my dear mothers wailing \n And the prayers of three brothers that Id see no more \n And I felt fathers tears as he begged for forgiveness \n For seeking a new life on the still distant shore. \n Over in Killarney \n Many years ago, \n Me Mither sang a song to me \n In tones so sweet and low. \n Just a simple little ditty, \n In her good ould Irish way, \n And ld give the world if she could sing \n  That song to me this day. \n Too-ra-loo-ra-loo-ral, Too-ra-loo-ra-li, \n Too-ra-loo-ra-loo-ral, hush now, dont you cry! \n Too-ra-loo-ra-loo-ral, thats an Irish lullaby. \n Oft in dreams I wander \n To that cot again, \n I feel her arms a-huggin me \n As when she held me then. \n And I hear her voice a -hummin \n To me as in days of yore, \n When she used to rock me fast asleep \nOutside the cabin door. \n And who are you, me pretty fair maid \n And who are you, me honey? \n She answered me quite  modestly: \n I am me mothers darling. \n  With me too-ry-ay \n Fol-de-diddle-day \n Di-re fol-de-diddle \n Dai-rie oh. \n And will you come to me mothers house, \n When the sun is shining clearly \n Ill open the door and Ill let you in \n And divil o one would hear us. \n So I went to her house in the middle of the night \n When the moon was shining clearly \n Shc opened the door and she let me in \n And divil the one did hear us. \n She took me horse by the bridle and the bit \n And she led him to the stable \n Saying Theres plenty of oats for a soldiers horse, \n To eat it if hes able. \n Then she took me by the lily-white hand \n And she led me to the table \n Saying: Theres plenty of wine for a soldier boy, \n To drink it if youre able. \n Then I got up and made the bed \n And I made it nice and aisy \n Then I got up and laid her down \n Saying: Lassie, are you able? \n And there we lay till the break of day \n And divil a one did hear us \n Then I arose and put on me clothes \n Saying: Lassie, I must leave you. \n And when will you return again \n And when will we get married \n When broken shells make Christmas bells."  """

# First I tried hard coding lines of training result. Now try reading entire file of training data 
data = open('poem.txt').read()
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
seed_text = "I've got a bad feeling about this"
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