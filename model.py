from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding, SpatialDropout1D, LSTM, Dense


def getLSTM(n_words = 80000, input_length = 400, embedding_dim = 200,n_outputs = 5):
    model = Sequential()
    model.add(Embedding(n_words, embedding_dim, input_length=input_length))
    # regularization ??
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(embedding_dim, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
