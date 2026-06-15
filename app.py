import pickle
import re
from flask import Flask, render_template, request

app = Flask(__name__)

print("Loading model and vectorizer...")

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model and vectorizer loaded successfully!")


def clean_text(text):
    """
    Applies the same text cleaning as done during training.
    This ensures consistency between training and prediction.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()                          # Lowercase the text
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation/special characters
    text = re.sub(r'\s+', ' ', text).strip()     # Normalize whitespace
    return text


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Home page route.
    - GET  : Renders the empty form.
    - POST : Accepts the message, predicts, and renders the result.
    """
    prediction = None   # No result yet
    message = None      # No message yet
    is_spam = None      # Used to control styling in the template

    if request.method == 'POST':
        # Get the text message submitted by the user from the form
        message = request.form.get('message', '').strip()

        if message:
            # Step 1: Clean the text (same preprocessing as during training)
            cleaned = clean_text(message)

            # Step 2: Transform text into numerical features using the loaded vectorizer
            features = vectorizer.transform([cleaned])

            # Step 3: Use the model to predict (returns 'spam' or 'ham')
            result = model.predict(features)[0]

            # Step 4: Get the confidence probabilities [ham_prob, spam_prob]
            proba = model.predict_proba(features)[0]
            confidence = round(max(proba) * 100, 2)

            # Step 5: Build a user-friendly result string
            if result == 'spam':
                prediction = f"🚨 SPAM — This message looks like spam!"
                is_spam = True
            else:
                prediction = f"✅ NOT SPAM — This message looks legitimate."
                is_spam = False

    # Render the template with the prediction result
    return render_template('index.html',
                           prediction=prediction,
                           message=message,
                           is_spam=is_spam)


if __name__ == '__main__':
    # debug=False is safer for demos; set host='0.0.0.0' so Docker can expose it
    app.run(host='0.0.0.0', port=5000, debug=False)
