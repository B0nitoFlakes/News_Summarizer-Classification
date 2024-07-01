from keras import backend as K
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model 
from keras.layers import Dense, TimeDistributed, LSTM, Embedding, Input, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

tf.config.set_visible_devices([], 'GPU')

class AttentionLayer(tf.keras.layers.Layer):
    """
    This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
    There are three sets of weights introduced W_a, U_a, and V_a
     """

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]



# Load the saved model
model = tf.keras.models.load_model('model.h5', custom_objects={'AttentionLayer': AttentionLayer})

# Load tokenizers
with open('news_tokenizer.pkl', 'rb') as f:
    news_tokenizer = pickle.load(f)

with open('headline_tokenizer.pkl', 'rb') as f:
    headline_tokenizer = pickle.load(f)

max_len_news = 53

# Add '<START>' and '<END>' to the vocabulary of headline_tokenizer
start_token_index = len(headline_tokenizer.word_index) + 1
end_token_index = len(headline_tokenizer.word_index) + 2

headline_tokenizer.word_index['<START>'] = start_token_index
headline_tokenizer.index_word[start_token_index] = '<START>'

headline_tokenizer.word_index['<END>'] = end_token_index
headline_tokenizer.index_word[end_token_index] = '<END>'

news_vocab = len(news_tokenizer.word_index) + 1

headline_vocab = len(headline_tokenizer.word_index) + 1

K.clear_session()

embedding_dim = 300 #Size of word embeddings.
latent_dim = 500 #No. of neurons in LSTM layer.

encoder_input = Input(shape=(max_len_news, ), name='encoder_input')
encoder_emb = Embedding(news_vocab, embedding_dim, trainable=True, name='encoder_embedding')(encoder_input) #Embedding Layer

#Three-stacked LSTM layers for encoder. Return_state returns the activation state vectors, a(t) and c(t), return_sequences return the output of the neurons y(t).
#With layers stacked one above the other, y(t) of previous layer becomes x(t) of next layer.
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)
y_1, a_1, c_1 = encoder_lstm1(encoder_emb)

encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)
y_2, a_2, c_2 = encoder_lstm2(y_1)

encoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)
encoder_output, a_enc, c_enc = encoder_lstm3(y_2)

#Single LSTM layer for decoder followed by Dense softmax layer to predict the next word in summary.
decoder_input = Input(shape=(None,), name='decoder_input')
decoder_emb = Embedding(headline_vocab, embedding_dim, trainable=True, name='decoder_embedding')(decoder_input)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)
decoder_output, decoder_fwd, decoder_back = decoder_lstm(decoder_emb, initial_state=[a_enc, c_enc]) #Final output states of encoder last layer are fed into decoder.

#Attention Layer
attn_layer = AttentionLayer(name='attention_layer') 
attn_out, attn_states = attn_layer([encoder_output, decoder_output]) 

decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_output, attn_out])

decoder_dense = TimeDistributed(Dense(headline_vocab, activation='softmax'))
decoder_output = decoder_dense(decoder_concat_input)

model = Model([encoder_input, decoder_input], decoder_output)



#Encoder inference model with trained inputs and outputs.
encoder_model = Model(inputs=encoder_input, outputs=[encoder_output, a_enc, c_enc])

#Initialising state vectors for decoder.
decoder_initial_state_a = Input(shape=(latent_dim,))
decoder_initial_state_c = Input(shape=(latent_dim,))
decoder_hidden_state = Input(shape=(max_len_news, latent_dim))

#Decoder inference model
decoder_out, decoder_a, decoder_c = decoder_lstm(decoder_emb, initial_state=[decoder_initial_state_a, decoder_initial_state_c])
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state, decoder_out])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_out, attn_out_inf])

decoder_final = decoder_dense(decoder_inf_concat)
decoder_model = Model([decoder_input]+[decoder_hidden_state, decoder_initial_state_a, decoder_initial_state_c], [decoder_final]+[decoder_a, decoder_c])



def decoded_sequence(input_seq):
    encoder_out, encoder_a, encoder_c = encoder_model.predict(input_seq)  # Collecting output from encoder inference model.
    
    # Initialize input to decoder neuron with START token.
    next_input = np.zeros((1, 1))
    next_input[0, 0] = headline_tokenizer.word_index['<START>']
    
    output_seq = ''
    
    # Set a maximum limit for the number of tokens in the generated summary to avoid infinite loops
    max_summary_length = 14
    current_length = 0
    
    while current_length < max_summary_length:
        # Output from decoder inference model, with output states of encoder used for initialization.
        decoded_out, trans_state_a, trans_state_c = decoder_model.predict([next_input] + [encoder_out, encoder_a, encoder_c])
        
        # Get index of output token from y(t) of decoder.
        output_idx = np.argmax(decoded_out[0, -1, :])
        
        # If output index corresponds to END token, summary is terminated without adding the END token itself.
        if headline_tokenizer.index_word.get(output_idx) == '<END>':
            break
        
        # Generate the token from the index.
        output_token = headline_tokenizer.index_word.get(output_idx, '')  
        
        # Skip the '<START>' token in the generated sequence
        if output_token != '<START>':
            output_seq = output_seq + ' ' + output_token  # Append to summary
            current_length += 1
        
        # Pass the current output index as input to the next neuron.
        next_input[0, 0] = output_idx
        
        # Continuously update the transient state vectors in the decoder.
        encoder_a, encoder_c = trans_state_a, trans_state_c
    
    return output_seq


def generate_summary(news_text):
    input_sequence = news_tokenizer.texts_to_sequences([news_text])
    input_sequence_padded = pad_sequences(input_sequence, maxlen=max_len_news, padding='post')
    predicted_summary = decoded_sequence(input_sequence_padded)
    return predicted_summary

#==========================================================================================
# News Identification

import joblib
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer

# Load the logistic regression model
lr_model = joblib.load('logistic_regression_model.joblib')

loaded_cv = joblib.load('count_vectorizer.joblib')

#Remove Tags
def remove_tags(text):
  remove = re.compile(r'<.*?>')
  return re.sub(remove, '', text)

#Special Character removal
def special_char(text):
  reviews = ''
  for x in text:
    if x.isalnum():
      reviews = reviews + x
    else:
      reviews = reviews + ' '
  return reviews

#Lowercasing
def convert_lower(text):
   return text.lower()

#Stopwords removal
def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  words = word_tokenize(text)
  return [x for x in words if x not in stop_words]

#Lemmatization
def lemmatize_word(text):
  wordnet = WordNetLemmatizer()
  return " ".join([wordnet.lemmatize(word) for word in text])

def preprocess_text(text):
    text = remove_tags(text)
    text = special_char(text)
    text = convert_lower(text)
    text = remove_stopwords(text)
    text = lemmatize_word(text)
    return text


def classify_news(news_input):
    
    processed_input = preprocess_text(news_input)

    y_pred1 = loaded_cv.transform([processed_input])

    # Make predictions using the logistic regression model
    yy = lr_model.predict(y_pred1)

    result = ""
    if yy == [1]:
        result = "Sports News"
    elif yy == [2]:
        result = "Business News"
    elif yy == [3]:
        result = "Wellness News"
    elif yy == [4]:
        result = "Politics News"
    elif yy == [5]:
        result = "Entertainment News"
    elif yy == [6]:
        result = "Travel News"
    elif yy == [7]:
        result = "Style and Beauty News"
    elif yy == [8]:
        result = "Parenting News"
    elif yy == [9]:
        result = "Food and Drink News"
    elif yy == [10]:
        result = "World News"
    elif yy == [11]:
        result = "Technology News"
    elif yy == [12]:
        result = "Science News"
    elif yy == [13]:
        result = "Automobile News"
    return result

#=================================================================================
#Fine tune PEGASUS
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

def generate_summary_pegasus(text):
    # Load the model and tokenizer
    model = PegasusForConditionalGeneration.from_pretrained("fine_tuned_pegasus")
    tokenizer = PegasusTokenizer.from_pretrained("fine_tuned_pegasus")

    # Set the device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    # Generate predictions for the input text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)

    # Decode the output tokens to get the predicted text
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

#=================================================================

from transformers import pipeline

def generate_summary_t5_pipe(news_input):
    pipe_t5 = pipeline('summarization', model='t5-small')
    pipe_out_t5 = pipe_t5(news_input)
    # Extract summary text from the dictionary
    summary_text_t5 = pipe_out_t5[0]['summary_text']
    
    return summary_text_t5

def generate_summary_bart_pipe(news_input):
    pipe_bart = pipeline("summarization", model="facebook/bart-large-cnn")
    pipe_out_bart = pipe_bart(news_input)
    # Extract summary text from the dictionary
    summary_text_bart = pipe_out_bart[0]['summary_text']
    
    return summary_text_bart

def generate_summary_pegasus_pipe(news_input):
    pipe_pegasus = pipeline("summarization", model="google/pegasus-xsum")
    pipe_out_pegasus = pipe_pegasus(news_input)
    # Extract summary text from the dictionary
    summary_text_pegasus = pipe_out_pegasus[0]['summary_text']
    
    return summary_text_pegasus

# Streamlit app
def main():
    st.title("Text Summarization App")

    # User input for news text
    news_input = st.text_area("Enter the news text:")

    if st.button("Summarize"):
        if news_input:
            # Generate and display the summary
            summary = generate_summary(news_input)
            summary_pegasus_finetune = generate_summary_pegasus(news_input)
            summary_t5 = generate_summary_t5_pipe(news_input)
            summary_bart = generate_summary_bart_pipe(news_input)
            summary_pegasus = generate_summary_pegasus_pipe(news_input)

            classification_result = classify_news(news_input)

            st.subheader("News Classification:")
            st.write("Predicted Class:", classification_result)
            st.subheader("LSTM")
            st.write(summary)
            st.subheader("PEGASUS-FINETUNED")
            st.write(summary_pegasus_finetune)
            st.subheader("PEGASUS")
            st.write(summary_pegasus)
            st.subheader("T5")
            st.write(summary_t5)
            st.subheader("BART")
            st.write(summary_bart)
        else:
            st.warning("Please enter some news text.")

if __name__ == "__main__":
    main()
