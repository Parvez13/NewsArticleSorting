from flask import Flask, request, render_template
from flask_cors import cross_origin
import tensorflow as tf
import logging
import json
from keras_preprocessing.text import tokenizer_from_json
from data_preprocessing import preprocess_text

logging.basicConfig(filename='All_logs/app.log',
                    filemode='a',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

# Load the tensorflow gpu
logging.info('Get the Tensorflow GPU')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

# Load saved "cnn" model for prediction
logging.info('Load saved "CNN" model')
model = tf.keras.models.load_model('Model/CNN1d.h5')
output_seq_len = 593

# Open "tokenizer" for our raw data
logging.info("Open 'tokenizer' as json to our raw data")
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
@cross_origin()
def predict():
    if request.method == 'POST':
        input_message = request.form['message']
        message = preprocess_text(str(input_message))
        my_input = [message]
        input_sequence = tokenizer.texts_to_sequences(my_input)
        input_pad = tf.keras.utils.pad_sequences(input_sequence, padding='post', maxlen=output_seq_len)
        pred_probs = model.predict(input_pad)
        logging.info('Get the predictions')
        preds = tf.argmax(pred_probs, axis=1)
        if preds == [0]:
            logging.info("Our model predicted this news as Business")
            prediction = 'This news is related to "BUSINESS"'
        elif preds == [1]:
            logging.info("Our model predicted this news as Entertainment.")
            prediction = 'This news is related to "ENTERTAINMENT"'
        elif preds == [2]:
            logging.info("Our model predicted this news as Politics.")
            prediction = 'This news is related to "POLITICS"'
        elif preds == [3]:
            logging.info("Our model predicted this news as Sports.")
            prediction= 'This news is related to "SPORTS"'
        elif preds == [4]:
            logging.info("Our model predicted this news as Technology.")
            prediction = 'This news is related to "TECHNOLOGY"'
        else:
            prediction = "This is an anonymous news"

        return render_template('home.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

