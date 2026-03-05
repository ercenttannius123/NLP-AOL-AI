from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# load model dan vectorizer
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return "Email Spam Detection API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    email_text = data["text"]

    # ubah text ke vector
    vector = tfidf.transform([email_text])

    # prediksi
    prediction = model.predict(vector)[0]

    if prediction == 1:
        result = "SPAM"
    else:
        result = "NOT SPAM"

    return jsonify({
        "email": email_text,
        "prediction": result
    })


if __name__ == "__main__":
    print("Email Spam Detection API Running")
    app.run(debug=True)