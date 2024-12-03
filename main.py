import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

# Load and preprocess the text
filepath = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(filepath, 'rb').read().decode(encoding='utf-8').lower()
text = text[300000:800000]  # Selecting a part of the text for training
characters = sorted(set(text))

# Create dictionaries for character-to-index and index-to-character mappings
char_to_index = {c: i for i, c in enumerate(characters)}
index_to_char = {i: c for i, c in enumerate(characters)}

SEQ_LENGTH = 40
STEP_SIZE = 3

'''
# Prepare the dataset
sentences = []
next_characters = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

# Convert to numpy arrays
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)
y = np.zeros((len(sentences), len(characters)), dtype=bool)

for i, sentence in enumerate(sentences):
    for t, character in enumerate(sentence):
        x[i, t, char_to_index[character]] = 1
    y[i, char_to_index[next_characters[i]]] = 1

# Build the RNN model
model = Sequential()
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))
model.add(Dense(len(characters)))
model.add(Activation('softmax'))

# Compile the model
optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Train the model
model.fit(x, y, batch_size=256, epochs=4)

# Save the model
model.save('textgenerator.model.keras')

'''

model = tf.keras.models.load_model('textgenerator.model.keras')

#takes prediction of our model and makes a choice depends on temperature (high: experimental, low : safe pick)
#this function picks one character
def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1,preds,1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) - SEQ_LENGTH - 1)
    generated_text = ''
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated_text += sentence

    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):  # Encode the sentence (not the full character set)
            x[0, t, char_to_index[character]] = 1
        
        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, temperature)
        next_character = index_to_char[next_index]

        generated_text += next_character
        sentence = sentence[1:] + next_character  # Slide the window to the right

    return generated_text


print("-----------0.2------------")
print(generate_text(300,0.2))
print("-----------0.4------------")
print(generate_text(300,0.4))
print("-----------0.6------------")
print(generate_text(300,0.6))
print("-----------0.8------------")
print(generate_text(300,0.8))
print("-----------1.0------------")
print(generate_text(300,1.0))