#Training sequence models in Keras requires:
  #To use mini-batches of same length by padding them with zeros.
  #An Embedding() layer can be initialized with pretrained values which can fixed or trainable.
      #If the dataset is small then making this layer trainable is not worthy.
  #LSTM() has a flag called return_sequences.
      #If return_sequences = True then it will return every hidden state.
      #If return_sequences = False then it will return only the last one.
  #To Regularize your network you can use Dropout() right after LSTM()

#Import Dependencies
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

#Pad the Sentences with zero vectors to have the same length as that of the length of the maximum sequence.

def sentences_to_indices(X, word_to_index, max_len):
  m = X.shape[0]
  X_indices = np.zeros((m, max_len))
  
  for i in range(m):
    sentence_words = [w.lower() for w in X[i].split()]
    j = 0
    
    for w in sentence_words:
      X_indices[i, j] = word_to_index[w]
      j += 1
  
  return X_indices

X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
print("X1 =", X1)
print("X1_indices =", X1_indices)

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
  vocab_len = len(word_to_index) + 1
  
  emb_dim = word_to_vec_map["solid"].shape[0]
  
  emb_matrix = np.zeros((vocab_len, emb_dim))
  
  for word, index in word_to_index.items():
    emb_matrix[index, :] = word_to_vec_map[word]
    
  embedding_layer = Embedding(vocab_len, emb-dim, trainable = False)
  
  embedding_layer.build((None, ))
  
  embedding_layer.set_weights([emb_matrix])
  
  return embedding_layer

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
  sentence_indices = Input(input_shape, dtype = 'int32')
  
  embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
  embeddings = embedding_layer(sentence_indices)
  
  X = LSTM(128, return_sequences = True)(embeddings)
  X = Dropout(0.5)(X)
  X = LSTM(128, return_sequences = False)(X)
  X = Dropout(0.5)(X)
  X = Dense(5)(X)
  X = Activation("softmax")(X)
  
  model = Model(inputs = sentence_indices, outputs = X)
  
  return model

model = Emojify_V2((maxLen, ), word_to_vec_map, word_to_index)
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)

model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle = True)

X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc)

pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    num = np.argmax(pred[i])
    
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num))

X_test_example = np.array(['Great but not Exciting'])
print(X_test_example[0] + ' ' + label_to_emoji(np.argmax(model.predict(X_test_indices))))

        
   
                     
