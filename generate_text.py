#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 22:05:54 2018

@author: ladvien
"""
def generate(model, sentence, maxlen, chars, char_indices, sample, diversity, indices_char):
    import sys
    import numpy as np
    generated = ''
    generated += sentence
    
    sentence = sentence.lower()
    
    for i in range(400):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_indices[char]] = 1.
    
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]
    
        generated += next_char
        sentence = sentence[1:] + next_char
    
        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()