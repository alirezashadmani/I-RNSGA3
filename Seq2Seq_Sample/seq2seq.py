# download spanish to english data

import os, unicodedata, re, io
import tensorflow as tf

path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin = 'http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract = True
)

path_to_file = os.path.dirname(path_to_zip) + '/spa-eng/spa.txt'

# convert unicode files to ascii

def unicode_to_ascii(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):

    w = unicode_to_ascii(w.lower().strip())
    # creating a space between a word and the punctuation following it
    # w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r"([.,!?¿])", r" \1 ", w)
    w = re.sub('\s{2,}', ' ', w)

    # replacing everything with space, except letters, punctuations, ....
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
    w = w.strip()

    # adding start and eng tokens
    w = '<start> ' + w + ' <end>'
    return w

# create dataset

def create_dataset(path, num_examples):

    lines = io.open(path, encoding = 'UTF-8'.read().strip().split('\n'))
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

    return zip(*word_pairs)

# tokenize the sentence and pad the sequence to the same length
def tokenize(lang):

    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filter = '')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocesing.sequence.pad_sequences(tensor, padding = 'post')
    return tensor, lang_tokenizer

def load_dataset(path, num_examples = None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer