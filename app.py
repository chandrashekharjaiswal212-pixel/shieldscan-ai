from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import pandas as pd
import io
import os
from pypdf import PdfReader

app = Flask(__name__)
CORS(app)

# Load trained 98.8% model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# 1. Frontend Web Page Serve karne ke liye
@app.route("/")
def home():
    return send_file("index.html")

# 2. Single Message Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message is empty"}), 400

    transformed_text = vectorizer.transform([message])
    prediction = model.predict(transformed_text)[0]
    
    classes = list(model.classes_)
    spam_index = classes.index("spam")
    probabilities = model.predict_proba(transformed_text)[0]
    spam_probability = round(float(probabilities[spam_index]) * 100, 2)

    return jsonify({
        "status": "success",
        "prediction": prediction,
        "is_spam": bool(prediction == "spam"),
        "spam_probability": spam_probability
    })

# 3. Bulk CSV / PDF Scanner Endpoint
@app.route("/bulk-predict", methods=["POST"])
def bulk_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    filename = file.filename.lower()

    if not (filename.endswith('.csv') or filename.endswith('.pdf')):
        return jsonify({"error": "Only CSV and PDF files are supported"}), 400

    try:
        messages = []

        if filename.endswith('.pdf'):
            reader = PdfReader(file.stream)
            full_pdf_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    full_pdf_text += "\n" + text
            
            raw_lines = [line.strip() for line in full_pdf_text.split('\n') if len(line.strip()) > 8]
            if not raw_lines:
                return jsonify({"error": "PDF me koi readable text nahi mila!"}), 400
            messages = raw_lines

        elif filename.endswith('.csv'):
            content = file.stream.read().decode("utf-8", errors="ignore")
            df = pd.read_csv(io.StringIO(content))
            
            possible_cols = ['message', 'text', 'sms', 'content', 'v2', 'body']
            found_col = None
            for col in df.columns:
                if str(col).strip().lower() in possible_cols:
                    found_col = col
                    break
            
            if not found_col:
                found_col = df.columns[0]

            messages = df[found_col].dropna().astype(str).tolist()

        if not messages:
            return jsonify({"error": "File me koi messages nahi mile!"}), 400

        transformed = vectorizer.transform(messages)
        predictions = model.predict(transformed)
        probs = model.predict_proba(transformed)

        classes = list(model.classes_)
        spam_idx = classes.index("spam")

        results = []
        spam_count = 0
        ham_count = 0

        for i, text in enumerate(messages):
            is_spam = (predictions[i] == 'spam')
            prob = round(float(probs[i][spam_idx]) * 100, 2)
            if is_spam:
                spam_count += 1
            else:
                ham_count += 1
                
            results.append({
                "id": i + 1,
                "message": text[:80] + ("..." if len(text) > 80 else ""),
                "full_message": text,
                "is_spam": bool(is_spam),
                "spam_probability": prob
            })

        return jsonify({
            "status": "success",
            "total": len(messages),
            "spam_count": spam_count,
            "ham_count": ham_count,
            "results": results[:50]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    app.run(host="0.0.0.0", port=port)