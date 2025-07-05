import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Sample data 
texts = [
    "I love this movie",
    "This film was terrible",
    "Amazing performance",
    "I hate this",
    "I really enjoyed it",
    "Worst experience ever",
    "Fantastic!",
    "Not good at all"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

max_words = 1000  
tokenizer = keras.preprocessing.text.Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

maxlen = 10
X = keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)


y = np.array(labels)

model = keras.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=16, input_length=maxlen),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


history = model.fit(X, y, epochs=10, verbose=1)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

test_sentences = [
    "I absolutely loved it",
    "I don't like it at all"
]
test_seq = tokenizer.texts_to_sequences(test_sentences)
test_pad = keras.preprocessing.sequence.pad_sequences(test_seq, maxlen=maxlen)

predictions = model.predict(test_pad)
for sentence, pred in zip(test_sentences, predictions):
    print(f"{sentence} -> {'Positive' if pred[0] > 0.5 else 'Negative'}")
