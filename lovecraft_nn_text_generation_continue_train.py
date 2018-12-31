'''

'''

from __future__ import print_function
import numpy as np

workpath = '/home/ladvien/nn_lovecraft'
save_model_path = '/home/ladvien/nn_lovecraft/data/models'
corpus_path = workpath + '/data/lovecraft_corpus.txt'

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def load(save_model_path, model_filename, weights_filename):
    from keras.models import model_from_json
    
    # Load model from JSON.
    json_file = open(save_model_path + '/' + model_filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    # Load weights from H5.
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(save_model_path + '/' + weights_filename)
    return loaded_model

model = load(save_model_path, model_filename = 'model2018-12-24 13:41.json', weights_filename = 'weights_2018-12-24 13:39.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam')
