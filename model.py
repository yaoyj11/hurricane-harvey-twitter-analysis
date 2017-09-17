'''Trains an LSTM model on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:
  - RNNs are tricky. Choice of batch size is important,
  choice of loss and optimizer is critical, etc.
  Some configurations won't converge.
  - LSTM loss decrease patterns during training can be quite different
  from what you see with CNNs/MLPs/etc.
  '''
  from __future__ import print_function

  from keras.preprocessing import sequence
  from keras.models import Sequential
  from keras.layers import Dense, Embedding
  from keras.layers import LSTM
  from keras.datasets import imdb


def seqModel(config):
  model = Sequential()
  model.add(Embedding(config.max_features, config.embedding))
  model.add(LSTM(config.embedding, dropout=0.2, recurrent_dropout=0.2))
  model.add(Dense(1, activation='sigmoid'))
