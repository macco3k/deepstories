import os
import re
import numpy as np
import glob
# import seq2seq

from string import punctuation
from itertools import islice
from nltk import corpus, stem
from gensim.models import KeyedVectors
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Masking, Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical

DATA_PATH = 'data'
OUTPUT_PATH = 'output'

punct = set(punctuation)

w2v = KeyedVectors.load_word2vec_format(os.path.join(DATA_PATH, 'GoogleNews-vectors-negative300.bin'), binary=True)

def expand_contractions(text, w2v):
    cont = Contractions(w2v_model=w2v)
    return cont.expand_texts(text=text, precise=True)

vocab_dim = w2v.vector_size
eos_vector = np.ones((vocab_dim))
unk_vector = np.zeros((vocab_dim))

def preprocess(text):
    text = re.sub(repl=' ', string=text, pattern='-')
    return re.sub(repl='', string=text, pattern='[{}\n\t\\\\]'.format(''.join(punctuation)))

def build_vocabulary(file_list, w2v):
    idx2word = set() # only count unique words
    missing_words = set()
    for file in file_list:
        print('build_vocabulary: processing [{}]'.format(file))
        with open(file, 'r', encoding='utf-8') as f:
            for i,line in enumerate(f):
                line = preprocess(line)
                if len(line) == 0:
                    print('Line {} is empty. Skipping it.'.format(i+1))
                    continue
                
                for word in line.split(' '):
                    # skip words without embeddings. They'll be assigned the <UNK> token
                    if len(word) > 0:
                        if word in w2v:
                            idx2word.add(word)
                        else:
                            missing_words.add(word)
                        
    missing_words = sorted(list(missing_words))
    idx2word = sorted(list(idx2word))
    idx2word.insert(0, '<EOS>')
    idx2word.insert(1, '<UNK>')
    word2idx = {w:i for i,w in enumerate(idx2word)}
    # skip EOS and UNK when looking up word embeddings
    word2embeddings = {**{'<EOS>': eos_vector, '<UNK>': unk_vector}, **{w:w2v[w] for w in idx2word[2:]}}
    return idx2word, word2idx, word2embeddings, missing_words

def prepare_data(file_list, word2idx):
    vocab_size = len(word2idx)    
    data = []
    for file in file_list:
        print('prepare_data: processing [{}]'.format(file))
        with open(file, 'r', encoding='utf-8') as f:
            file_data = []
            for i,line in enumerate(f):
                line = preprocess(line)
                if len(line) == 0:
                    print('Line {} is empty. Skipping it.'.format(i+1))
                    continue
                # return the integer representation of the sentence
                file_data.append([word2idx[w] if w in word2idx else word2idx['<UNK>'] for w in line.split(' ')])
        data.append(file_data)
    return data
                                 
def get_embedding_matrix(word2embeddings):
    embedding_dim = len(list(word2embeddings.values())[0])
    embedding_matrix = np.zeros(shape=(len(word2embeddings), embedding_dim))
    for i, w in enumerate(word2embeddings):
        embedding_matrix[i] = word2embeddings[w]
    return embedding_matrix
                                
def prepare_input(input_text, word2embeddings):
    return [word2embeddings[word] if word in word2embeddings else unk_vector for word in preprocess(input_text).split(' ') if len(word) > 0]


file_list = glob.glob('data/parsed/*.txt')
idx2word, word2idx, word2embeddings, missing_words = build_vocabulary(file_list, w2v)
data = prepare_data(file_list, word2idx)
vocab_size = len(idx2word)

embedding_matrix = get_embedding_matrix(word2embeddings)


# Define a batch generator
class BatchGenerator(object):            
    def __init__(self, data, batch_size=1):
        self.data = data
        self.batch_size = batch_size
        self.UNK = word2idx['<UNK>']
        self.EOS = word2idx['<EOS>']
        self.PAD = 0
        self.eye = np.eye(len(word2idx))
        
    def generate_batch(self): 
        def window(seq, n=3, step=1):
            "Returns a sliding window (of width n) over data from the iterable"
            "   s -> (s[0],...s[n-1]), (s[0+skip_n],...,s[n-1+skip_n]), ...   "
            it = iter(seq)
            result = tuple(islice(it, n))
            if len(result) == n:
                yield result    

            result = result[step:]
            for elem in it:
                result = result + (elem,)
                if len(result) == n:
                    yield result
                    result = result[step:]
                    
        def to_categorical(sentence):
            return [self.eye[wordidx] for wordidx in sentence]
                    
        # every three lines comprise a sample sequence where the first two items
        # are the input and the last one is the output
        i  = 1 # batch counter        
        x_enc = []
        x_dec = []
        y  = []
        while True:
            for play in self.data:
                j  = 1 # sample counter
                for scene, command, reply in window(play, n=3, step=2):
                    scene_command = scene + command
                    
                    encoder_input  = np.array(scene_command + [self.EOS])
                    decoder_input  = np.array(reply)
                    decoder_output = np.array(to_categorical(reply[1:] + [self.EOS]))
                    
                    #print(encoder_input.shape, decoder_input.shape, decoder_output.shape)
                
                    x_enc.append(encoder_input)
                    x_dec.append(decoder_input)
                    y.append(decoder_output)
                    if i == self.batch_size or j == len(play):
                        if self.batch_size > 1:
                            # pad and return the batch
                            x_enc = sequence.pad_sequences(x_enc, padding='post', value=self.PAD)
                            x_dec = sequence.pad_sequences(x_dec, padding='post', value=self.PAD)
                            y     = sequence.pad_sequences(y,     padding='post', value=self.PAD) 

                        x_out, y_out = [np.array(x_enc.copy()), np.array(x_dec.copy())], np.array(y.copy())

                        i  = 1
                        x_enc = []
                        x_dec = []
                        y  = []

                        yield (x_out, y_out)
                    else:
                        i += 1 # next sample per batch
                    j += 1 # next sample
                    
            # no more data, just stop the generator
            break
            
            
generator = BatchGenerator(data, batch_size=16)
sample = next(generator.generate_batch())


# returns train, inference_encoder and inference_decoder models
def define_models(src_vocab_dim, dst_vocab_dim=None, latent_dim=300, mask_value=0, embedding_matrix=None):
    # define training encoder. We use return_state to retrieve the hidden states for the encoder and
    # provide them as input to the decoder
    if dst_vocab_dim is None:
        dst_vocab_dim = src_vocab_dim
        
    encoder_inputs = Input(shape=(None,)) # timesteps, features (one-hot encoding)
    encoder_masking = Masking(mask_value=mask_value)(encoder_inputs)
    
    if embedding_matrix is not None:
        encoder_masking = Embedding(input_dim=src_vocab_dim, output_dim=latent_dim, weights=[embedding_matrix], 
                                   trainable=False)(encoder_masking)
        
    encoder = LSTM(units=latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_masking)
    encoder_states = [state_h, state_c]
    
    # define training decoder. It is initialized with the encoder hidden states
    decoder_inputs = Input(shape=(None,))
    decoder_masking = Masking(mask_value=mask_value)(decoder_inputs)
    
    if embedding_matrix is not None:
        decoder_masking = Embedding(input_dim=src_vocab_dim, output_dim=latent_dim, weights=[embedding_matrix], 
                                   trainable=False)(decoder_masking)
    
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_masking, initial_state=encoder_states)
    decoder_dense = Dense(dst_vocab_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # define inference decoder
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_masking, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    # return all models
    return model, encoder_model, decoder_model


model, encinf, decinf = define_models(src_vocab_dim=vocab_size, embedding_matrix=embedding_matrix)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.summary(line_length=110)

batch_generator = BatchGenerator(data, batch_size=32)
model.fit_generator(batch_generator.generate_batch(), steps_per_epoch=1000, epochs=25)

model.save('data/')