#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 23:49:00 2018

@author: ladvien
"""
import io
import re
import random
import keras
import tensorflow as tf

############################################
# Data preperation and Training Parameters #
############################################
retain_sentence_length_threshold = 6
min_head_length = 5
number_of_corpus_samples = 3
word_freq_threshold_for_commonizing = 1

if retain_sentence_length_threshold < min_head_length:
    print('The sentence length must be greater than the minimum head length')

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
                new_sentence += 'LFW'
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
workpath = '/home/ladvien/nn_lovecraft'
save_model_path = '/home/ladvien/nn_lovecraft/data/models'
corpus_path = workpath + '/data/lovecraft_corpus.txt'

with io.open(corpus_path, encoding='utf-8') as f:
    text = f.read().lower()
    

########################################
# Sentences                            #
########################################
    
# Clear special characters.
text = clean_special_chars(text)

# Split into sentences by '.', ',', '!', ';', or '?'
sentences = re.split(r'[.,!?;]', text)

# Remove blank sentences.
for sentence in sentences:
    num_words = len(sentence.split(' '))
    if num_words <= retain_sentence_length_threshold:
        sentences.remove(sentence)

# Preseverse import strings.
for i in range(len(sentences)):
    sentences[i] = sentences[i].strip()

# Get hte frequency of words
word_freqs = get_words_and_frequencies(text)

# Divide the cleaned corpus into sentences
sentences = commonize_low_freq_words(sentences, word_freqs, word_freq_threshold_for_commonizing)

# Get a list of distinct words.
distinct_words = list(word_freqs.keys())

#################################
# Get Sentence Heads and Butts  #
#################################
heads = []
butts = []

for _ in range(number_of_corpus_samples):
    # Split sentence into words
    for sentence in sentences:
        sent_word_list = sentence.split(' ')
        sentence_len = len(sent_word_list)
        
        # Make sure there are enough words in sentence to create a head and butt.
        if sentence_len > min_head_length:
            # Split the sentence at a random index.
            split_index = random.randint(min_head_length,sentence_len)
            heads.append(' '.join(sent_word_list[0:split_index]))
            butts.append(' '.join(sent_word_list[split_index:sentence_len]))


text = ''
for i in range(len(heads)):
    text += heads[i]
    text += butts[i]
    text = text.strip()
    text += '. '

#################################
# Tokenize Heads and Butts      #
#################################
tokenizer = keras.preprocessing.text.Tokenizer()

# Include model signals in the token set.
special_tokens = ['SOS', 'EOS', 'LFW']
sentences  = sentences + special_tokens
tokens = tokenizer.fit_on_texts(sentences)
tokens = tokenizer.word_index

# Tokenize the heads and butt lists.
tokenized_heads = tokenizer.texts_to_sequences(heads)
tokenized_butts = tokenizer.texts_to_sequences(heads)

#####################################################################################################
# MODEL                                                                                             #
#####################################################################################################

# Create placeholders for inputs and targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'inputs')
    targets = tf.placeholder(tf.int32, [None, None], name = 'targets')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob

# Preprocessing the targets
def preprocess_targets(targets, tokens, batch_size):
    left_side = tf.fill([batch_size, 1], tokens['sos'])
    right_side = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1,1]) 
    preprocessed_targets = tf.concat([left_side, right_side], axis = 1)
    return preprocessed_targets

def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rn_BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state

def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_state = tf.zeroes([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                             attention_keys,
                                                                             attention_values,
                                                                             attention_score_function,
                                                                             attention_construct_function,
                                                                             name = 'attn_dec_train')
    
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)   
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

def decode_test_set(encoder_state, decoder_cell, decoder_embedded_input, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_state = tf.zeroes([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_state,
                                                                                                                                    attention_option = 'bahdanau',
                                                                                                                                    num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(
                                                                             encoder_state[0],
                                                                             attention_keys,
                                                                             attention_values,
                                                                             attention_score_function,
                                                                             attention_construct_function,
                                                                             name = 'attn_dec_train')
    
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)   
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)