import collections

import helper
import numpy as np
import project_tests as tests

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, LSTM, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
import os
from tensorflow.python.client import device_lib
from keras.models import Model, Sequential
from keras.models import load_model

def load_dataSeparete(path):
    """
    Load Dataset from File
    """
    data = []
    input_file = os.path.join(path)

    with open(input_file) as my_file:
        for line in my_file:
            lineD = line
            #print(lineD+'  Farshad')
            data.append(lineD)
    #input_file = os.path.join(path)
    #with open(input_file, 'r', encoding='utf-8') as f:
        #data = f.read()
    
    return data


def load_data():
    # Load English data
    sourceLanguage = load_dataSeparete('data/small_vocab_en')
    # Load French data
    targetLanguage = load_dataSeparete('data/small_vocab_fr')    
    print('Dataset Loaded')
    for sample_i in range(2):
        print('small_vocab_en Line {}:  {}'.format(sample_i + 1, sourceLanguage[sample_i]))
        print('small_vocab_fr Line {}:  {}'.format(sample_i + 1, targetLanguage[sample_i]))
    
    return sourceLanguage, targetLanguage
    
    
    
def statsOnData(sourceLanguage, targetLanguage):
    english_words_counter = collections.Counter([word for sentence in sourceLanguage for word in sentence.split()])
    french_words_counter = collections.Counter([word for sentence in targetLanguage for word in sentence.split()])
    
    print('{} English words.'.format(len([word for sentence in sourceLanguage for word in sentence.split()])))
    print('{} unique English words.'.format(len(english_words_counter)))
    print('10 Most common words in the English dataset:')
    print('"' + '" "'.join(list(zip(*english_words_counter.most_common(10)))[0]) + '"')
    print()
    print('{} French words.'.format(len([word for sentence in targetLanguage for word in sentence.split()])))
    print('{} unique French words.'.format(len(french_words_counter)))
    print('10 Most common words in the French dataset:')
    print('"' + '" "'.join(list(zip(*french_words_counter.most_common(10)))[0]) + '"')
    
    
def tokenize(x):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    t = Tokenizer()
    t.fit_on_texts(x)
    return t.texts_to_sequences(x), t


def pad(x, length=None):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    # TODO: Implement
    if not length:
        length = max([len(sent) for sent in x])
    return pad_sequences(x, maxlen=length, padding='post')

def preprocess(x, y):
    """
    Preprocess x and y
    :param x: Feature List of sentences
    :param y: Label List of sentences
    :return: Tuple of (Preprocessed x, Preprocessed y, x tokenizer, y tokenizer)
    """
    preprocess_x, x_tk = tokenize(x)
    preprocess_y, y_tk = tokenize(y)

    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)

    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)

    return preprocess_x, preprocess_y, x_tk, y_tk


def logits_to_text(logits, tokenizer):
    """
    Turn logits from a neural network into text using the tokenizer
    :param logits: Logits from a neural network
    :param tokenizer: Keras Tokenizer fit on the labels
    :return: String that represents the text of the logits
    """
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])



def moreStatsOnData(l1, l2):
    preproc_L1_sentences, preproc_L2_sentences, L1_tokenizer, L2_tokenizer =\
    preprocess(l1, l2)
    
    max_L1_sequence_length = preproc_L1_sentences.shape[1]
    max_L2_sequence_length = preproc_L2_sentences.shape[1]
    L1_vocab_size = len(L1_tokenizer.word_index)
    L2_vocab_size = len(L2_tokenizer.word_index)
    
    print('Data Preprocessed')
    print("Max English sentence length:", max_L1_sequence_length)
    print("Max French sentence length:", max_L2_sequence_length)
    print("English vocabulary size:", L1_vocab_size)
    print("French vocabulary size:", L2_vocab_size)
    return preproc_L1_sentences, preproc_L2_sentences, L1_tokenizer, L2_tokenizer, L1_vocab_size, L2_vocab_size



def embed_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    rnn = GRU(64, return_sequences=True, activation="tanh")
    
    embedding = Embedding(french_vocab_size, 64, input_length=input_shape[1]) 
    logits = TimeDistributed(Dense(french_vocab_size, activation="softmax"))
    
    model = Sequential()
    #em can only be used in first layer --> Keras Documentation
    model.add(embedding)
    model.add(rnn)
    model.add(logits)
    model.compile(loss=sparse_categorical_crossentropy,
                  optimizer=Adam(learning_rate),
                  metrics=['accuracy'])
    
    return model

def bd_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
   
    learning_rate = 1e-3
    model = Sequential()
    model.add(Bidirectional(GRU(128, return_sequences = True, dropout = 0.1), 
                           input_shape = input_shape[1:]))
    model.add(TimeDistributed(Dense(french_vocab_size, activation = 'softmax')))
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    return model

def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences = True)(input_seq)
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)
    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    
    return model

def encdec_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  
    learning_rate = 1e-3
    model = Sequential()
    model.add(GRU(128, input_shape = input_shape[1:], return_sequences = False))
    model.add(RepeatVector(output_sequence_length))
    model.add(GRU(128, return_sequences = True))
    model.add(TimeDistributed(Dense(french_vocab_size, activation = 'softmax')))
    
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    return model


def model_final(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
  
    model = Sequential()
    model.add(Embedding(input_dim=english_vocab_size,output_dim=128,input_length=input_shape[1]))
    model.add(Bidirectional(GRU(256,return_sequences=False)))
    model.add(RepeatVector(output_sequence_length))
    model.add(Bidirectional(GRU(256,return_sequences=True)))
    model.add(TimeDistributed(Dense(french_vocab_size,activation='softmax')))
    learning_rate = 0.005
    
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    
    return model


def final_predictions(x, y, x_tk, y_tk):
    """
    Gets predictions using the final model
    :param x: Preprocessed English data
    :param y: Preprocessed French data
    :param x_tk: English tokenizer
    :param y_tk: French tokenizer
    """
    # TODO: Train neural network using model_final
    model = model_final(x.shape, y.shape[1], len(x_tk.word_index)+1, len(y_tk.word_index)+1)
    model.fit(x,y,batch_size=1024, epochs=25,validation_split=0.2)
    model.save('my_model.h5')
    
    
def translateThis(model, txts, x, y, x_tk, y_tk):

    ## DON'T EDIT ANYTHING BELOW THIS LINE
    y_id_to_word = {value: key for key, value in y_tk.word_index.items()}
    y_id_to_word[0] = '<PAD>'

    sentence = txts
    sentence = [x_tk.word_index[word] for word in sentence.split()]
    sentence = pad_sequences([sentence], maxlen=x.shape[-1], padding='post')
    sentences = np.array([sentence[0], x[0]])
   
    
    predictions = model.predict(sentence, len(sentence))

    print('Sample 1:')
    print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[0]]))
    #print('Il a vu un vieux camion jaune')
    print('Sample 2:')
    #print(' '.join([y_id_to_word[np.argmax(x)] for x in predictions[1]]))
    #print(' '.join([y_id_to_word[np.max(x)] for x in y[0]]))

l1, l2 = load_data()
#statsOnData(l1, l2)

p_L1, p_L2, L1_token, L2_token, L1_size, L2_size   =moreStatsOnData(l1, l2)

final_predictions(p_L1, p_L2, L1_token, L2_token)

# load YAML and create model
model = load_model('my_model.h5')
print("Loaded model from disk")
translateThis(model,'paris is never cold during october', p_L1, p_L2, L1_token, L2_token)
    