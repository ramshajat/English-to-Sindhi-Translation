from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import numpy as np
import keras_nlp
import numpy as np
import pandas as pd
import pathlib
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import  model_from_json
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
app = Flask(__name__)
app.config['SECRET_KEY'] = "abcmsdihgiosjawfjeioghaegnklaebn"
BATCH_SIZE = 128
EPOCHS = 15  # This should be at least 10 for convergence
MAX_SEQUENCE_LENGTH = 20
ENG_VOCAB_SIZE = 104890
SND_VOCAB_SIZE = 104890

EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8


@app.route('/')
def index():
    return render_template('input_form.html')



@app.route('/results', methods=['POST'])
def transate():
    text = request.form['text']
    # result= algo(text)
    return render_template('results.html', predicted_text=text)


if __name__ == '__main__':
    app.run(debug=True)