# importing libraries
import streamlit as st
import pandas as pd
import scipy
from scipy import spatial
from sentence_transformers import SentenceTransformer
import ktrain



# get_text is a simple function to get user input from text_input
def get_text():
    input_text = st.text_input("You: ","type here")
    return input_text


# # data input
df=pd.read_csv("Cleaned_Chatbot_data.csv")
p = ktrain.load_predictor('farmer_model')

#bot responses
def chatbot_response(msg):
    df = pd.read_csv("Cleaned_Chatbot_data.csv")
    p = ktrain.load_predictor('C:\\Users\\Garlapati.Varun\\PycharmProjects\\Farmer_bot\\farmer_model')
    model = ktrain.get_predictor(p.model, p.preproc)
    msg=msg.lower()
    print("Message is",msg)
    greetings = ['hi', 'hey', 'hello', 'heyy', 'hi', 'hey', 'good evening', 'good morning', 'good afternoon', 'good',
             'fine', 'okay', 'great', 'could be better', 'not so great', 'very well thanks', 'fine and you',
             "i'm doing well", 'pleasure to meet you', 'hi whatsup']
    goodbyes = ['thank you', 'thank you', 'yes bye', 'bye', 'thanks and bye', 'ok thanks bye', 'goodbye', 'see ya later',
            'alright thanks bye', "that's all bye", 'nice talking with you', 'i’ve gotta go', 'i’m off', 'good night',
            'see ya', 'see ya later', 'catch ya later', 'adios', 'talk to you later', 'bye bye', 'all right then',
            'thanks', 'thank you', 'thx', 'thx bye', 'thnks', 'thank u for ur help', 'many thanks', 'you saved my day',
            'thanks a bunch', "i can't thank you enough", "you're great", 'thanks a ton', 'grateful for your help',
            'i owe you one', 'thanks a million', 'really appreciate your help', 'no', 'no goodbye']
    if msg in greetings:
        suitable_answer = "Hi! I\'m a Farmers chatbot!.My name is Kisan.Please ask me for help whenever you feel like it! I\'m always online."
    elif msg in goodbyes:
        suitable_answer = "Hope I was able to help you today! Take care, bye!"
    else:
        zero_data = df.dropna(subset=['length_question'])
        zero_data = zero_data.dropna(subset=['clean_text'])
        questions = [str(msg)]
        intent = model.predict(questions[0])
        print("Intent is :", intent)
        intent_df = zero_data[zero_data["clean_type"] == intent]
        responses = intent_df["KCCAns"].to_list()
        response_contexts = intent_df["QueryText"].to_list()
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("Model Downloaded")
        question_embeddings = model.encode(questions)
        print("Question Embedding Completed")
        response_embeddings = model.encode(response_contexts)
        print("Responses Embedding Completed")
        distances = []
        for i in response_embeddings:
            distances.append(1 - scipy.spatial.distance.cosine(question_embeddings, i))
            suitable_answer = responses[distances.index(max(distances))]
    return suitable_answer

st.sidebar.title("Farmer Bot - Kisan ")
st.title("""Kisan Bot helps Farmers in need """)


user_input = get_text()
if str(user_input) =="type here":
    response = "Please type your problem here"
else:
    response = chatbot_response(user_input)
st.text_area("Bot:", value=response, height=200, max_chars=None, key=None)