from tensorflow.python.framework import ops
import tflearn
import numpy as np
import pickle
import nltk
import json
from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS
import joblib
import random

stemmer = nltk.stem.lancaster.LancasterStemmer()
# model_file = 'predModel.pkl'
# with open(model_file, 'rb') as f:
#     predModel = pickle.load(f)

with open("intents.json") as file:
    data = json.load(file)

with open("model.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return np.array(bag)

app = Flask(__name__)
CORS(app)

@app.route('/get', methods=['GET'])
def get_bot_response():
    # global seat_count
    message =request.args.to_dict()
    if 'msg' in message:
        message = message['msg'].lower()
        results = model.predict([bag_of_words(message, words)])[0]
        result_index = np.argmax(results)
        tag = labels[result_index]
        if results[result_index] > 0.7:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
            response = random.choice(responses)
        else:
            response = "I didn't quite get that, please try again."
        return jsonify(response=response), 200, {'Content-Type': 'application/json'}
    return jsonify({"Missing Data!"})




imagePredModel = joblib.load('imagePredModel.joblib')


@app.route('/imagePred',methods=['POST'])
def imgPred():
    
    try:
        data = request.get_json()
        hex_color = data['color']
        numeric_color = hex_to_numeric(hex_color)
        probability = imagePredModel.predict_proba([[numeric_color]])[0][1] * 100
        result = '{:.2f}%'.format(probability)
        response = {
            'probability': result
        }

        return jsonify(response)
    except Exception as err:
        print(err)
        return jsonify({'message': 'Error'}), 500


def hex_to_numeric(hex_color):
    hex_color = hex_color.lstrip('#')
    return int(hex_color, 16)



predModel = joblib.load('predModel.joblib')
feature_names = ['age', 'gender', 'symptomsA', 'symptomsB', 'symptomsC', 'symptomsD', 'symptomsE',
                 'symptomsF', 'symptomsG', 'symptomsH', 'symptomsI', 'symptomsJ', 'symptomsK',
                 'symptomsL', 'symptomsM', 'symptomsN', 'symptomsO', 'durationSymptoms',
                 'sexContacts', 'condomUseAtLastSex', 'hivTest']

@app.route('/pred', methods=['POST'])
def predict():
    try:
        incoming_predictions = request.get_json()
        print("Data", incoming_predictions)

        # Prepare the user input for prediction
        user_data = incoming_predictions
        user_input = [
            int(user_data['age']),
            int(user_data['gender']),
            int(user_data['checkboxValues']['symptomA']),
            int(user_data['checkboxValues']['symptomB']),
            int(user_data['checkboxValues']['symptomC']),
            int(user_data['checkboxValues']['symptomD']),
            int(user_data['checkboxValues']['symptomE']),
            int(user_data['checkboxValues']['symptomF']),
            int(user_data['checkboxValues']['symptomG']),
            int(user_data['checkboxValues']['symptomH']),
            int(user_data['checkboxValues']['symptomI']),
            int(user_data['checkboxValues']['symptomJ']),
            int(user_data['checkboxValues']['symptomK']),
            int(user_data['checkboxValues']['symptomL']),
            int(user_data['checkboxValues']['symptomM']),
            int(user_data['checkboxValues']['symptomN']),
            int(user_data['checkboxValues']['symptomO']),
            int(user_data['duration']),
            int(user_data['sexContacts']),
            int(user_data['isUsedComdoms']),
            int(user_data['isTestHiv'])
        ]

        # Create a DataFrame with the user input and feature names
        user_df = pd.DataFrame([user_input], columns=feature_names)

        # Make a prediction using the loaded model
        prediction = predModel.predict(user_df)[0]

        # Prepare the response
        response = {
            "diagnosis": int(prediction),
            "userData": user_data
        }

        return jsonify({'predictions': response})
    except Exception as err:
        print(err)
        return jsonify({'message': 'Error'}), 500

if __name__ == '__main__':
    app.run()