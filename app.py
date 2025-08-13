import torch
import torch.nn as nn
import numpy as np
import spacy
from sklearn.preprocessing import LabelEncoder
import requests
from thefuzz import process
import os
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request, jsonify

# --- 1. Initialize Flask App ---
app = Flask(__name__)

# --- 2. Constants and Setup ---
# IMPORTANT: Paste your API Key here!
API_KEY = "YOUR_API_KEY_GOES_HERE" 

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("Model loaded.")

KNOWN_CITIES = ["Tokyo", "London", "Berlin", "Cairo", "New York", "Phoenix", "Paris"]

# --- 3. Load Knowledge Base ---
knowledge_base = []
try:
    with open("knowledge.txt", "r", encoding="utf-8") as f:
        for paragraph in f.read().split("\n\n"):
            if paragraph.strip():
                knowledge_base.append(paragraph.strip())
    kb_vectors = np.array([nlp(paragraph).vector for paragraph in knowledge_base])
    print("--- Knowledge base loaded and vectorized. ---")
except FileNotFoundError:
    print("--- knowledge.txt not found. Skipping knowledge base feature. ---")
    knowledge_base = []

# --- 4. Training Data ---
training_data = [
    ("Hello", "greeting"), ("Hi there", "greeting"), ("Good morning", "greeting"),
    ("Goodbye", "goodbye"), ("See you later", "goodbye"), ("Have a great day", "goodbye"), ("quit", "goodbye"),
    ("What's the weather like?", "weather"), ("Tell me the weather forecast", "weather"), ("Is it raining outside?", "weather"),
    ("Tell me a joke", "joke"), ("I want to hear a joke", "joke"), ("you are funny", "joke"),
    # New data for our knowledge base
    ("What is a synapse?", "question"),
    ("Tell me about signal flow", "question"),
    ("Explain Flask to me", "question"),
    ("What is sentiment analysis?", "question")
]

# --- 5. Data Processing ---
sentences = [item[0] for item in training_data]
labels = [item[1] for item in training_data]
X = np.array([nlp(text).vector for text in sentences])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# --- 6. Model Definition ---
class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(IntentClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.layer1(x); out = self.relu(out); out = self.layer2(out)
        return out

# --- 7. Model Training & Loading ---
input_size = X.shape[1]; hidden_size = 32; output_size = len(label_encoder.classes_)
model = IntentClassifier(input_size, hidden_size, output_size)
MODEL_SAVE_PATH = "model_state.pth"

if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    print("--- Trained model loaded from file. Skipping training. ---")
else:
    print("\n--- No saved model found. Starting Training ---")
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        outputs = model(X_tensor)
        loss = loss_function(outputs, y_tensor)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    print("--- Training Finished ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"--- Model saved to {MODEL_SAVE_PATH} ---")

# --- 8. Helper Functions ---
def get_weather(city="Phoenix"):
    base_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=imperial"
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        main = data['main']; weather_desc = data['weather'][0]['description']; temp = main['temp']
        return f"The current temperature in {city} is {temp}°F with {weather_desc}."
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 404: return f"Sorry, I couldn't find the city '{city}'. Please check the spelling."
        else: return f"Sorry, there was an HTTP error: {http_err}"
    except Exception: return f"An unexpected error occurred while fetching weather for {city}."

def predict_intent(sentence):
    sentence_vector = nlp(sentence).vector
    sentence_tensor = torch.tensor(sentence_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(sentence_tensor)
    _, predicted_index = torch.max(output, dim=1)
    return label_encoder.inverse_transform([predicted_index.item()])[0]

def answer_question_from_kb(question):
    if not knowledge_base: return "I don't have a knowledge base loaded to answer questions from."
    question_vector = nlp(question).vector.reshape(1, -1)
    similarities = cosine_similarity(question_vector, kb_vectors)
    most_similar_index = np.argmax(similarities)
    return knowledge_base[most_similar_index]

# --- 9. Flask Routes ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=['POST'])
def chat():
    user_message = request.json['message']
    response_message = ""

    blob = TextBlob(user_message)
    sentiment = blob.sentiment
    intent = predict_intent(user_message)
    
    if intent == "question":
        response_message = answer_question_from_kb(user_message)
    elif intent == "greeting":
        if sentiment.polarity > 0.3:
            response_message = "Hello! You seem to be in a great mood. How can I assist you today?"
        else:
            response_message = "Hello! How can I help you today?"
    elif intent == "goodbye":
        if sentiment.polarity < -0.3:
            response_message = "I'm sorry if I couldn't help more. I hope your day gets better. Goodbye."
        else:
            response_message = "Goodbye! Have a great day."
    elif intent == "weather":
        doc = nlp(user_message)
        city = "Phoenix"
        for ent in doc.ents:
            if ent.label_ == 'GPE':
                city = ent.text
                break
        response_message = get_weather(city)
    elif intent == "joke":
        response_message = "Why did the scarecrow win an award? Because he was outstanding in his field!"
    else:
        response_message = "I'm not sure how to respond to that yet."

    return jsonify({'response': response_message})

# --- 10. Run the App ---
if __name__ == "__main__":
    app.run(debug=True)