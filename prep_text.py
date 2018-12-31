#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 07:13:02 2018

@author: ladvien
"""
from bs4 import BeautifulSoup
import re
root_path = '/home/ladvien/nn_lovecraft/data'
corpus_path = root_path + '/lovecraft_corpus.txt'

with open(corpus_path, 'a+') as corpus_file:
    import glob
    text_paths = glob.glob("/home/ladvien/nn_lovecraft/data/xhtml/*.xhtml")
    
    for text_path in text_paths:
        with open(text_path) as html:
            soup = BeautifulSoup(html)
            
            text = soup.get_text()
            # Removes the CSS tags left in the text.
            text = re.sub('/*<[^>]+>*/', '', text)
            paragraphs = text.split('\n')

            # Remove front matter, title, and date            
            paragraphs = paragraphs[10:]
            
            clean_text = ''
            for paragraph in paragraphs:
                clean_text += paragraph

            '''
            Need to convert:
                This is a new world.Something else
            '''

            corpus_file.write(clean_text)
            