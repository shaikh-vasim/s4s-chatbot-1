import pickle
import json5 as json
import random2 as random
# import tensorflow as tf
import tflearn
import numpy as np
from flask import Flask, render_template, request, jsonify
import nltk
# import datetime
from nltk.stem.lancaster import LancasterStemmer
import COVID19Py


stemmer = LancasterStemmer()
seat_count = 50


with open("intents.json") as file:
    data = json.load(file)
with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

# tf.reset_default_graph()


net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get')
def get_bot_response():
    message = request.args.get('msg')
    if message:
        message = message.lower()
        results = model.predict([bag_of_words(message, words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]
        if results[result_index] > 0.7:
            if tag == 'covid19':
                covid19 = COVID19Py.COVID19(data_source='jhu')
                country = input('Enter Location...')

                if country.lower() == 'world':

                    latest_world = covid19.getLatest()
                    print(
                        'Confirmed:', latest_world['confirmed'], ' Deaths:', latest_world['deaths'])

                else:

                    latest = covid19.getLocations()

                    latest_conf = []
                    latest_deaths = []
                    for i in range(len(latest)):

                        if latest[i]['country'].lower() == country.lower():
                            latest_conf.append(
                                latest[i]['latest']['confirmed'])
                            latest_deaths.append(latest[i]['latest']['deaths'])
                    latest_conf = np.array(latest_conf)
                    latest_deaths = np.array(latest_deaths)
                    print('Confirmed: ', np.sum(latest_conf),
                          'Deaths: ', np.sum(latest_deaths))

            else:
                for tg in data['intents']:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                response = random.choice(responses)
        else:
            response = "I didn't quite get that, please try again."
        return str(response)
    return "Missing Data!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

    # app.run()
