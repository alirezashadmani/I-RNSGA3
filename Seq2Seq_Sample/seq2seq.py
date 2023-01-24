# download spanish to english data

import os, unicodedata, re, io, time
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


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):

        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras_layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequence = True,  # Whether to return the last output in the output sequence, or the full sequence.
                                       return_state = True, # Whether to return the last state in addition to the output.
                                       recurrent_initializer = 'glorot_uniform')
        
    def call(self, x, hidden):

        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state
    
    def initialize_hidden_state(self):

        return tf.zeros((self.batch_sz, self.enc_units))
        


class Decoder(tf.keras.Model):

    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):

        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequence = True,
                                       return_state = True,
                                       recurrent_initializer = 'glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, hidden):

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x, initial_state = hidden)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state
    
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True, reduction = 'none')

def loss_function(real, pred):

    loss_ = loss_object(real, pred)
    return tf.reduce_mean(loss_)


optimizer = tf.keras.optimizers.Adam()

def get_train_step_function():

    @tf.function
    def train_step(inp, targ, enc_hidden, encoder, decoder):

        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)

            dec_hidden = enc_hidden
            
            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

            for t in range(1, targ.shape[1]):
                predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                loss += loss_function(targ[:, t], predictions)
                dec_output = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss
    return train_step

def calculate_validation_loss(inp, targ, enc_hidden, encoder, decoder):

    loss = 0
    enc_output, enc_hidden = encoder(inp, enc_hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    for t in range(1, targ.shape[1]):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        loss += loss_function(targ[:, t], predictions)
        dec_input = tf.expand_dims(targ[:, t], 1)

    loss = loss / int(targ.shape[1])
    
    return loss

def training_seq2seq(epochs):

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    train_step_func = get_train_step_function()
    training_loss = []
    validation_loss = []

    for epoch in range(epochs):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step_func(inp,
                                         targ,
                                         enc_hidden,
                                         encoder,
                                         decoder)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss))

        enc_hidden = encoder.initialize_hidden_state()
        total_val_loss = 0

        for (batch, (inp, targ)) in enumerate(validation_dataset.take(steps_per_epoch_val)):
            val_loss = calculate_validation_loss(inp,
                                                 targ,
                                                 enc_hidden,
                                                 encoder,
                                                 decoder)
            total_val_loss += val_loss

        training_loss.append(total_loss / steps_per_epoch)
        validation_loss.append(total_val_loss / steps_per_epoch_val)

        print('Epoch {} Loss {:.4f} Validation Loss {:.4f}'.format(epoch + 1, training_loss[-1], validation_loss[-1]))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    return encoder, decoder, training_loss, validation_loss
