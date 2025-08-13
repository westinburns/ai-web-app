from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import spacy
from sklearn.preprocessing import LabelEncoder
import requests
from thefuzz import process
import os

# --- Initialize Flask App ---
app = Flask(__name__)

# --- All setup code from our previous project ---
# Make sure to paste your API Key here!
API_KEY = "32b291920fff3166010c7ece34b0323b"

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("Model loaded.")

KNOWN_CITIES = ["Tokyo", "London", "Berlin", "Cairo", "New York", "Phoenix", "Paris"]
training_data = [
    # Your training data from before goes here...
    ("Hello", "greeting"), ("Hi there", "greeting"),
    ("Goodbye", "goodbye"), ("See you later", "goodbye"),
    ("What's the weather like?", "weather"), ("Tell me the weather forecast", "weather"),
    ("Tell me a joke", "joke"), ("I want to hear a joke", "joke"),
]

sentences = [item[0] for item in training_data]
labels = [item[1] for item in training_data]
X = np.array([nlp(text).vector for text in sentences])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# --- Model Definition ---
class IntentClassifier(nn.Module):
    # ... (Model class code from before)
    def __init__(self, input_size, hidden_size, output_size):
        super(IntentClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.layer1(x); out = self.relu(out); out = self.layer2(out)
        return out

# --- Training and Loading Logic ---
input_size = X.shape[1]; hidden_size = 32; output_size = len(label_encoder.classes_)
model = IntentClassifier(input_size, hidden_size, output_size)
MODEL_SAVE_PATH = "model_state.pth"

if os.path.exists(MODEL_SAVE_PATH):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    print("--- Trained model loaded from file. ---")
else:
    print("--- Training new model... ---")
    # ... (Training loop code from before)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(100):
        outputs = model(torch.tensor(X, dtype=torch.float32))
        loss = loss_function(outputs, torch.tensor(y, dtype=torch.long))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("--- Model trained and saved. ---")


# --- Helper Functions (No changes here) ---
def get_weather(city="Phoenix"):
    # ... (Your final get_weather function)
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
    except Exception as err:
        return f"An unexpected error occurred: {err}"

def predict_intent(sentence):
    # ... (Your predict_intent function)
    sentence_vector = nlp(sentence).vector
    sentence_tensor = torch.tensor(sentence_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(sentence_tensor)
    _, predicted_index = torch.max(output, dim=1)
    return label_encoder.inverse_transform([predicted_index.item()])[0]


# --- Flask Routes ---

@app.route("/")
def home():
    return render_template("index.html")

# THIS IS THE UPDATED CHAT ROUTE
@app.route("/chat", methods=['POST'])
def chat():
    user_message = request.json['message']
    response_message = ""

    # --- NEW: Sentiment Analysis Step ---
    blob = TextBlob(user_message)
    sentiment = blob.sentiment # Gets polarity and subjectivity
    
    # Predict the primary intent
    intent = predict_intent(user_message)
    
    # --- UPDATED: Action logic now considers sentiment ---
    if intent == "greeting":
        if sentiment.polarity > 0.3: # If user sounds positive
            response_message = "Hello! You seem to be in a great mood. How can I assist you today?"
        else:
            response_message = "Hello! How can I help you today?"

    elif intent == "goodbye":
        if sentiment.polarity < -0.3: # If user sounds negative
            response_message = "I'm sorry if I couldn't help more. I hope your day gets better. Goodbye."
        else:
            response_message = "Goodbye! Have a great day."

    elif intent == "weather":
        # (Weather logic from before, no changes needed)
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

# --- Run the App ---
if __name__ == "__main__":
    # (Make sure your model loading/training logic is here)
    app.run(debug=True)