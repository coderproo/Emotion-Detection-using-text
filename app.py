from flask import Flask, request, render_template
import pickle
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()
app = Flask(__name__)

# Load components
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        raw_text = request.form["text"]
        text = clean_text(raw_text)
        vector = vectorizer.transform([text])
        pred = model.predict(vector)[0]
        emotion = encoder.inverse_transform([pred])[0]
        prediction = f"Detected Emotion: {emotion.capitalize()}"
    return render_template("index.html", prediction=prediction)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
