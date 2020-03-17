import discord
import time
import asyncio
import nltk
import sys

nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle

with open('intents.json') as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

# with open("data.pickle", "wb") as f:
# pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    sdlfk
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)


# model.save('model.tflearn')

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = (1)

    return np.array(bag)


# def chat():
#     print("Start talking with the bot! (type quit to stop)")
#     while True:
#         inp = input("You: ")
#         if inp.lower() == 'quit':
#             break
#
#         results = model.predict([bag_of_words(inp, words)])[0]
#         results_index = np.argmax(results)
#         tag = labels[results_index]
#
#         if results[results_index] > 0.8:
#             for tg in data['intents']:
#                 if tg['tag'] == tag:
#                     responses = tg['responses']
#             print(random.choice(responses))
#         else:
#             print("I didn't understand. Try again.")


#id = 688977036832014364
messages = joined = 0

def read_token():
    with open('token.txt', 'r') as f:
        lines = f.readlines()
        return lines[0].strip()

token = read_token()

client = discord.Client()

async def update_stats():
    await client.wait_until_ready()
    global messages, joined

    while not client.is_closed():
        try:
            with open ('stats.txt', 'a') as f:
                f.write(f"Time: {int(time.time())}, Messages: {messages}, Members Joined: {joined}\n")

            messages = 0
            joined = 0

            await asyncio.sleep(5)
        except Exception as e:
            print(e)
            await asyncio.sleep(5)


@client.event
async def on_member_join(member):
    global joined
    joined += 1
    for channel in member.server.channels:
        if str(channel) == 'general':
            await client.send(f'''Welcome to the server {member.mention}''')


@client.event
async def on_message(message):
    global messages
    messages += 1

    print("message was sent")
    id = client.get_guild(688977036832014364)
    channels = ['general']
    valid_users = ["Rishi Peddakama#8261", "blubber#2381", "SlowWifi#0607"]

    # inp = message.content
    #
    # results = model.predict([bag_of_words(inp, words)])[0]
    # results_index = np.argmax(results)
    # tag = labels[results_index]
    #
    # print(results[results_index])
    # if results[results_index] > 0.8:
    #     for tg in data['intents']:
    #         if tg['tag'] == tag:
    #             responses = tg['responses']

    if str(message.channel) in channels and str(message.author) in valid_users:
        inp = message.content

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        print(results[results_index])
        if results[results_index] > 0.8:
            for tg in data['intents']:
                if tg['tag'] == tag:
                    responses = tg['responses']
        await message.channel.send(random.choice(responses))


    if str(message.content) == "!users":
        await message.channel.send(f"""There are {id.member_count} members""")

    if message.content.find("leave jarvis"):
        await message.channel.send("I do as u wish my master")
        sys.exit(-1)


client.loop.create_task(update_stats())
client.run(token)





