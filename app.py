# save this as app.py
from flask import Flask, request, render_template,  jsonify
import spacy
from spacy import displacy
import re
import json
import tflearn
import numpy
import random

import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.load("model.tflearn")

status_counter = 0

app = Flask(__name__)



@app.route("/")
def index():
    return render_template("unt.html")


@app.route("/get", methods=["GET","POST"])
def chat():
    global f
    msg = request.form["msg"]
    inp = msg
    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    # create or open a log file
    f= open("log.txt","a") 
    f.write("user: " + msg + "\n")

    global status_counter

    if status_counter == 0:
        if tag == "booking":
            status_counter += 1
            reply ="Sure, what is your departure and destination? PLease reply in the format of 'From City A to City B'"
            f.write("bot: " + reply + "\n")
            f.close()
            return reply
        else:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            reply = random.choice(responses)
            f.write("bot: " + reply + "\n")
            f.close()
            return reply
    else:
        if status_counter == 1:
            if check_formate_and_entity(inp) == True:
                status_counter += 1
                reply = "What about the date?"
                f.write("bot: " + reply + "\n")
                f.close()
                return reply
            else:
                reply = "Please reply in the format of 'from city A to City B"
                f.write("bot: " + reply + "\n")
                f.close()
                return reply

        if status_counter == 2:
            NER = spacy.load("en_core_web_sm")
            text1= NER(inp)
            count = 0
            for word in text1.ents:
                if word.label_ == "DATE":
                    count += 1
            if count > 0:
                status_counter += 1
                reply = "Can I have your name, please?"
                f.write("bot: " + reply + "\n")
                f.close()
                return reply
            else:
                reply = "Sorry, I couldn't understand. Could you tell me the date again?"
                f.write("bot: " + reply + "\n")
                f.close()
                return reply

        if status_counter == 3:
            NER = spacy.load("en_core_web_sm")
            text1= NER(inp)
            count = 0
            for word in text1.ents:
                if word.label_ == "PERSON":
                    count += 1
            if count > 0:
                status_counter += 1
                reply = "Can I have your email, please?"
                f.write("bot: " + reply + "\n")
                f.close()
                return reply
            else:
                reply = "Sorry, I didn't quit get it. Could you tell me your name again?"
                f.write("bot: " + reply + "\n")
                f.close()
                return reply

        if status_counter == 4:
            if isValid(inp):
                status_counter = 0
                reply = "Thank you. A link will send to your email to finish the following process."
                f.write("bot: " + reply + "\n")
                f.write("-----------a booking finished-----------\n")
                f.close()
                return reply
            else:
                reply = "It doesn't look like an email to me. Would you try it again?"
                f.write("bot: " + reply + "\n")
                f.close()
                return reply
  

def isValid(email):
    regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
    return re.search(regex, email)
  

def check_formate_and_entity(input):
    
    if "from " in input and " to " in input:
        cond1 = True
    else:
        cond1 = False

    NER = spacy.load("en_core_web_sm")
    text1= NER(input)
    count = 0
    for word in text1.ents:
        if word.label_ == "GPE":
            count += 1
    if count == 2:
        cond2 = True
    else:
        cond2 = False
    return cond1 and cond2

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)








if __name__ == "__main__":
   app.run()


