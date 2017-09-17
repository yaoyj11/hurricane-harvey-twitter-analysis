from reader import *


if __name__=="__main__":
  data=read_raw_data("./","small.csv",0,6)
  print data[0]
  max_features = 20000
  maxlen = 80  # cut texts after this number of words (among top max_features most common words)
  batch_size = 32

#  print('Loading data...')
#  (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
#  print(len(x_train), 'train sequences')
#  print(len(x_test), 'test sequences')
#
#  print('Pad sequences (samples x time)')
#  x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
#  x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
#  print('x_train shape:', x_train.shape)
#  print('x_test shape:', x_test.shape)
#
#
## try using different optimizers and different optimizer configs
#model.compile(loss='binary_crossentropy',
#                  optimizer='adam',
#                                metrics=['accuracy'])
#
#print('Train...')
#model.fit(x_train, y_train,
#              batch_size=batch_size,
#                        epochs=15,
#                                  validation_data=(x_test, y_test))
#score, acc = model.evaluate(x_test, y_test,
#                                batch_size=batch_size)
#print('Test score:', score)
#print('Test accuracy:', acc)
