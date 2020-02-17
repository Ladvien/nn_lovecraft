#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 23:49:00 2018

@author: ladvien
"""
from __future__ import absolute_import, division, print_function, unicode_literals

try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time
import random
import tensorflow.contrib.eager as tfe
tf.enable_eager_execution()


import gensim.downloader as api

##############
# References #
##############
# http://complx.me/2016-12-31-practical-seq2seq/
# https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb
# http://ruder.io/deep-learning-nlp-best-practices/index.html#bestpractices
# https://github.com/andreamad8/Universal-Transformer-Pytorch
# https://colab.research.google.com/github/tensorflow/docs/blob/r2.0rc/site/en/r2/tutorials/text/transformer.ipynb?authuser=1

##############
# Parameters #
##############

test_sentence           = '<sos> the darkness my old friend '
training_message        = 'Working'

num_examples            = 5000000
retain_threshold        = 15
min_perc_sent           = 0.4
max_perc_sent           = 0.6
corpus_samples          = 15
freq_threshold          = 1

max_sentence_len        = 600
min_sentence_len        = 40

epochs                  = 220
embedding_dim           = 1408
units                   = 1024
batch_size              = 16
attention_units         = 24
decoder_hidden_units    = 512
encoder_dropout         = 0.5
decoder_dropout         = 0.5
steps_per_epoch         = 100

split_sent_on           = r'[.!?]'

workpath                = '/home/ladvien/nn_lovecraft'
save_model_path         = '/home/ladvien/nn_lovecraft/data/models'
corpus_path             = workpath + '/data/lovecraft_corpus.txt'
output_filepath         = workpath + '/training_samples.txt'

#################
# Special Tokens
#################

start_of_sent           = '<sos>'  
end_of_sent             = '<eos>'
low_freq_word           = '<lfw>'

##################
# Load Embeddings
##################
print('Loading word vectors.')

# Load embeddings
#info = api.info()                       # show info about available models/datasets
embedding_model = api.load("glove-wiki-gigaword-300")    # download the model and return as object ready for use

vocab_size = len(embedding_model.vocab)

index2word = embedding_model.index2word
word2idx = {}
for index in range(vocab_size):
    word2idx[embedding_model.index2word[index]] = index


########################################
# Aid functions                        #
########################################

def clean_special_chars(text, convert_to_space = [], remove = []):
    
    ellipses = '<elp>'
    
    text = text.lower()
    
    # Artifact
    text = text.replace('return to table of contents', '')
    
    # Replace ellipses with token.
    text = text.replace('. . .', ellipses)
    text = text.replace('. . . .', ellipses)
    
    # Replaces new lines
    text = re.sub('\n', '', text)
    
    # Replaces multiple spaces
    text = re.sub(' +', ' ', text)
    
    # Replace handidness of quotations.
    text = re.sub(r'[“”"()]', '', text)
    
    # Opens parantheticals and speed-ups.
    text = re.sub('—', ' ', text)
    
    
    punctionation_marks = ['.', ',', '!', '?', ';', ':', '’s']
    
    for mark in punctionation_marks:
        text = text.replace(mark, ' ' + mark + ' ')
    
    # Replaces multiple spaces
    text = re.sub(' +', ' ', text)
    
    return text

def commonize_low_freq_words(sentences, word_frequencies, threshold, low_freq_word):
    
    print('')
    print(f'Replacing low-frequency words with {low_freq_word}')
    
    index = 0
    last_perc_comp = 0
    
    low_freq = []
    for key, value in word_frequencies.items():
        if value < freq_threshold:
            low_freq.append(key)
            
    clean_sentences = []
    for sentence in sentences:
        clean_sentence = ''
        for word in sentence.split(' '):
            if word in low_freq:
                clean_sentence += ' ' + low_freq_word
            else:
                clean_sentence += ' ' + word
        clean_sentences.append(clean_sentence)      
        index += 1
        
        perc_comp = int(round((index / len(sentences)) * 100, 2))
        if perc_comp % 10 == 0 and last_perc_comp < perc_comp:
            print(f'Complete: {str(perc_comp)}%')
            last_perc_comp = perc_comp

    return clean_sentences
    
# Get list of distinct words in string
def get_words_and_frequencies(text, delimiter = ' '):
    words = text.split(delimiter)
    word_frequencies = {}
    for word in words:
        word = re.sub('[^A-Za-z0-9]+', '', word)
        word = word.strip()   
        if word not in word_frequencies:
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
    return word_frequencies

########################################
# Load the Corpus                      #
########################################
with io.open(corpus_path, encoding='utf-8') as f:
    text = f.read().lower()
    

########################################
# Sentences                            #
########################################
    
# Clear special characters.
text = clean_special_chars(text)

# Split into sentences by '.', ',', '!', ';', or '?'
sentences = re.split(split_sent_on, text)

# Limit sentence size

new_sentences = []
for sentence in sentences:
    if len(sentence) < max_sentence_len:
        new_sentences.append(sentence)

sentences = new_sentences

# Limit the samples
sentences = sentences[0:num_examples]

# Add start and stop tokens.
sentences = [start_of_sent + ' ' + text + ' ' + end_of_sent for text in sentences]

# Remove blank sentences.
for sentence in sentences:
    num_words = len(sentence.split(' '))
    if num_words <= retain_threshold:
        sentences.remove(sentence)

# Preseverse import strings.
for i in range(len(sentences)):
    sentences[i] = sentences[i].strip()

# Get the frequency of words
word_freqs = get_words_and_frequencies(text)

# Divide the cleaned corpus into sentences
sentences = commonize_low_freq_words(sentences, word_freqs, freq_threshold, low_freq_word)

#################################
# Get Sentence Heads and Butts  #
#################################
heads = []
butts = []

for _ in range(corpus_samples):
    
    # Split sentence into words
    for sentence in sentences:
        sent_word_list = sentence.split(' ')
        sentence_len = len(sent_word_list)
        
        # Get split ratio
        split_index = int(sentence_len * random.uniform(min_perc_sent, max_perc_sent))  

        # Make sure there are enough words in sentence to create a head and butt.
        if sentence_len > split_index and split_index > min_sentence_len:
            # Split the sentence at a random index.
            heads.append(' '.join(sent_word_list[0:split_index]))
            butts.append(' '.join(sent_word_list[split_index:sentence_len + 1]))


text = ''
for i in range(len(heads)):
    text += heads[i]
    text += butts[i]
    text = text.strip()
    text += '. '

#################################
# Tokenize Heads and Butts      #
#################################
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')

# Include model signals in the token set.
special_tokens = [start_of_sent, end_of_sent, low_freq_word]
sentences  = sentences + special_tokens
tokenizer.fit_on_texts(sentences)
tokens = tokenizer.word_index

# Tokenize the heads and butt lists.
tokenized_heads = tokenizer.texts_to_sequences(heads)
tokenized_butts = tokenizer.texts_to_sequences(butts)

# Pad
tokenized_heads = tf.keras.preprocessing.sequence.pad_sequences(tokenized_heads, padding='pre')
tokenized_butts = tf.keras.preprocessing.sequence.pad_sequences(tokenized_butts, padding='post')

# Calculate max_length of the target tensors
def max_length(tensor):
    return max(len(t) for t in tensor)

max_length_heads, max_length_butts = max_length(tokenized_heads), max_length(tokenized_butts)

if max_length_butts != max_length_heads:
    print("Max lengths from the input sentences and output do not match.")

input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(tokenized_heads, tokenized_butts, test_size=0.2)

# Show length
print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))

def convert(lang, tensor):
  for t in tensor:
    if t != 0:
      print (f'{t:{5}} ----> {lang.index_word[t]}')

print ("Input Language; index to word mapping")
random_sent = random.randint(0, len(input_tensor_train))
convert(tokenizer, input_tensor_train[random_sent])
print ()
print ("Target Language; index to word mapping")
convert(tokenizer, target_tensor_train[random_sent])

########################
# MODEL Setup
########################
BUFFER_SIZE             = len(input_tensor_train)
vocab_inp_size          = len(tokenizer.word_index) + 1
vocab_tar_size          = len(tokenizer.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(batch_size, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape




#################
# Seq2Seq
#################
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, dropout = 0.5):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden, dropout = 0.5):
    x = self.embedding(x)
    x = self.dropout(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, batch_size, encoder_dropout)

# sample input
sample_hidden = encoder.initialize_hidden_state()
sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
print ('Encoder output shape: (batch size, sequence length, units) {}'.format(sample_output.shape))
print ('Encoder Hidden state shape: (batch size, units) {}'.format(sample_hidden.shape))

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # hidden shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # we are doing this to perform addition to calculate the score
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

attention_layer = BahdanauAttention(attention_units)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz, hidden, dropout = 0.5):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.dropout = tf.keras.layers.Dropout(dropout)
    self.hidden  = tf.keras.layers.Dense(hidden)
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output, dropout = True):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # dropout
    if dropout:
        output = self.dropout(output)
    
    output = self.hidden(output)
    
    if dropout:
        output = self.dropout(output)

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, batch_size, decoder_hidden_units, decoder_dropout)

sample_decoder_output, _, _ = decoder(tf.random.uniform((batch_size, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

#################
# Optimizer
#################

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

#################
# Save Best
#################
checkpoint_dir      = './training_checkpoints'
checkpoint_prefix   = os.path.join(checkpoint_dir, 'ckpt')
checkpoint          = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)


####################
# Train Functions
####################

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([tokenizer.word_index[start_of_sent]] * batch_size, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))
  print(type(batch_loss))
  return batch_loss


######################
# Evalulate Functions
######################
def evaluate(sentence):
    attention_plot = np.zeros((max_length_butts, max_length_heads))
    sentence = [x for x in sentence.split(' ') if x != '']
    sentence = ' '.join(sentence)
    inputs = [tokenizer.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_heads,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''
    
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden, dropout = 0.0)
    
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([tokenizer.word_index[start_of_sent]], 0)
    
    for t in range(max_length_butts):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out,
                                                             False)
    
        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()
    
        predicted_id = tf.argmax(predictions[0]).numpy()
    
        result += tokenizer.index_word[predicted_id] + ' '
    
        if tokenizer.index_word[predicted_id] == end_of_sent:
            return result, sentence, attention_plot
    
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
    
    return result, sentence, attention_plot

    
# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    
def generate(sentence, plot = False):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted: {}'.format(result))
    
    if plot:
        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        plot_attention(attention_plot, sentence.split(' '), result.split(' '))

    return result, sentence

def get_random_head(heads):
    return heads[random.randint(0, len(heads))]


######################
# Train
######################
with open(output_filepath, 'w+') as f:
    f.write(f'{training_message}\n\n')
    
for epoch in range(epochs):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 5 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))
        
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(round(time.time() - start), 2))
 
  # Test
  print('')
  print(f'Sample from epoch: {epoch}')
  with open(output_filepath, 'a') as f:
      # Save samples to file.
      f.write(f'Epoch: {epoch}, loss: {str(batch_loss.numpy())}\n')
      for _ in range(5): 
          head = get_random_head(heads)
          result, sentence = generate(head)
          f.write(f'I: {head}\n')
          f.write(f'O:{result}\n')        
      result, sentence = generate(test_sentence)
      f.write(f'TI: {test_sentence}\n')
      f.write(f'O : {result}\n')
      f.write('\n')
  print('')
        
  
  
######################
# Evalulate
######################
# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    

# Test 5 heads
for _ in range(5):
    generate(get_random_head(heads), plot = True)
    
def ghostwriter(text):
    # Test my own
    result, input_sentence = generate(text.lower().\
                                      replace('.', ' ' + end_of_sent)\
                                      .replace('\n', ' ')\
                                      .replace('\t', '')\
                                      .replace('\'s', ' ’s')\
                                      .replace(',', ' ,')
                                      )
    result = result.replace(' ' + end_of_sent + ' ', '.')
    input_sentence = input_sentence.replace(start_of_sent + ' ', '') + ' '
    full_sentence = input_sentence + result
    return full_sentence.capitalize()

ghostwriter(u""" I could not help myself, the madness was ensuing, 
            there was no end to all I had seen but I realised what my host had told me. 
            I carried on throughout the night absurdly ignoring his mother's place in my mind. 
            When my eye had never thought to watch for the delaying effect. 
            If not for the pictures of people, I would have felt tested and strangely alone.
            Though, some of the Mazurewicz were and dancing in the square of the yard for hours. 
            Pursuing a delicate end for their captives, while I wished for loathing of the fire 
            or to know where the choking room lie, since there had been little talk over
            their share in the chimney, which he was not known to appreciate. """)
