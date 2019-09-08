#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 23:49:00 2018

@author: ladvien
"""
import io
import re
import random
import os
import time

import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

##############
# Parameters #
##############

retain_threshold        = 6
min_perc_sent           = 0.2
max_perc_sent           = 0.7
corpus_samples          = 3
freq_threshold          = 1

max_sentence_len        = 300

workpath                = '/home/ladvien/nn_lovecraft'
save_model_path         = '/home/ladvien/nn_lovecraft/data/models'
corpus_path             = workpath + '/data/lovecraft_corpus.txt'

#################
# Special Tokens
#################
start_of_sent           = '<sos>'  
end_of_sent             = '<eos>'
low_freq_word           = '<lfw>'

########################################
# Aid functions                        #
########################################

def clean_special_chars(text, convert_to_space = [], remove = []):
    
    if len(convert_to_space) < 1:
        convert_to_space = ['..', '—', '--', ':']
        
    if len(remove) < 1:
        remove = ['"', '#', '$', '%', '&', '(', ')', '*', '+', '-', '/', '<', '=', '>', 
                   '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '’', '“', '”']
    
    # Replace sub-punctionations characters with space.
    rx = '[' + re.escape(''.join(convert_to_space)) + ']'
    text = re.sub(rx, ' ', text)

    # Remove non-alphanumeric
    rx = '[' + re.escape(''.join(remove)) + ']'
    text = re.sub(rx, '', text)
    return text

def commonize_low_freq_words(sentences, word_frequencies, threshold):

    bad_words = []
    for word in word_frequencies:
        if word_frequencies[word] < threshold:
            bad_words.append(word)
            print(word)
            
    if len(bad_words) < 1:
        return sentences
    
    new_sentences = []
    for sentence in sentences:
        if sentence == ' ':
            continue
        new_sentence = []
        for word in sentence.split(' '):
            if word in bad_words:
                new_sentence += low_freq_word
            else:
                new_sentence += word
            new_sentence += ' '
        new_sentences.append(''.join(new_sentence))
    
    sentences = new_sentences
    
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
sentences = re.split(r'[.,!?;]', text)

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

# Get hte frequency of words
word_freqs = get_words_and_frequencies(text)

# Divide the cleaned corpus into sentences
sentences = commonize_low_freq_words(sentences, word_freqs, freq_threshold)

# Get a list of distinct words.
distinct_words = list(word_freqs.keys())

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
        if sentence_len > split_index:
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
tokenizer = keras.preprocessing.text.Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')

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
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(tokenizer.word_index)+1
vocab_tar_size = len(tokenizer.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

example_input_batch, example_target_batch = next(iter(dataset))
example_input_batch.shape, example_target_batch.shape

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

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
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
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

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((64, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

checkpoint_dir      = './training_checkpoints'
checkpoint_prefix   = os.path.join(checkpoint_dir, 'ckpt')
checkpoint          = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([tokenizer.word_index[start_of_sent]] * BATCH_SIZE, 1)

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

  return batch_loss


EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))