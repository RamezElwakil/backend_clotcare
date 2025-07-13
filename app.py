from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import joblib
import numpy as np
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# === Flask App Setup ===
app = Flask(__name__, static_folder='../frontend', static_url_path='/')
CORS(app)

# === Email Configuration ===
try:
    from email_config import EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECIPIENT, ADDITIONAL_RECIPIENTS
except ImportError:
    # Fallback values if config file doesn't exist
    EMAIL_SENDER = "your-email@gmail.com"  # Replace with your Gmail
    EMAIL_PASSWORD = "your-app-password"   # Replace with your Gmail app password
    EMAIL_RECIPIENT = "your-email@gmail.com"  # Replace with your email to receive alerts
    ADDITIONAL_RECIPIENTS = []

# === Load ML Components ===
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === Email Function ===
def send_emergency_email():
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECIPIENT
        msg['Subject'] = "🚨 EMERGENCY: ClotCare Curing System Activated"
        
        # Email body
        body = f"""
        🚨 EMERGENCY ALERT 🚨
        
        The ClotCare Curing System has been activated!
        
        Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        This indicates a potential pulmonary embolism emergency.
        
        Please check the system immediately and take appropriate action.
        
        ---
        ClotCare Emergency Response System
        Automated Alert
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        text = msg.as_string()
        
        # Send to main recipient
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, text)
        
        # Send to additional recipients if any
        for recipient in ADDITIONAL_RECIPIENTS:
            if recipient.strip():  # Skip empty entries
                server.sendmail(EMAIL_SENDER, recipient.strip(), text)
        
        server.quit()
        
        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

# === Emergency Email Endpoint ===
@app.route('/send-emergency-email', methods=['POST'])
def trigger_emergency_email():
    success = send_emergency_email()
    if success:
        return jsonify({'message': 'Emergency email sent successfully'}), 200
    else:
        return jsonify({'error': 'Failed to send emergency email'}), 500

# === Predict Route ===
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400

    questions = [
        'age', 'trauma', 'vt_history', 'cancer', 'lung', 'renal', 'diabetes',
        'temperature', 'bmi', 'edema', 'immobility', 'pneumonia', 'platelets', 'af'
    ]

    try:
        input_features = []
        for q in questions:
            value = data.get(q)
            if value is None:
                return jsonify({'error': f'Missing value for: {q}'}), 400

            if q in ['age', 'temperature', 'bmi', 'platelets']:
                input_features.append(float(value))
            else:
                if value not in ['Yes', 'No']:
                    return jsonify({'error': f'Invalid value for {q}: {value}. Must be Yes or No.'}), 400
                input_features.append(1 if value == 'Yes' else 0)

        input_array = np.array([input_features])
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)
        probabilities = model.predict_proba(input_scaled)
        pe_probability = round(float(probabilities[0][1]) * 100, 2)

        # Determine risk level
        if pe_probability < 35:
            risk_level = "Low Risk"
        elif pe_probability < 50:
            risk_level = "Medium Risk"
        elif pe_probability < 75:
            risk_level = "High Risk"
        else:
            risk_level = "Severe Risk"

        return jsonify({
            'message': f'The person has a {pe_probability}% chance of developing PE in the future.',
            'risk_level': risk_level
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === Static Frontend ===
STATIC_FOLDER = app.static_folder or os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend'))

@app.route('/')
def index():
    return send_from_directory(STATIC_FOLDER, 'index.html')

@app.route('/form')
def form():
    return send_from_directory(STATIC_FOLDER, 'form.html')

@app.route('/monitor')
def monitor():
    return send_from_directory(STATIC_FOLDER, 'monitor.html')

@app.route('/curing')
def curing():
    return send_from_directory(STATIC_FOLDER, 'curing.html')

@app.route('/chatbot')
def chatbot():
    return send_from_directory(STATIC_FOLDER, 'chatbot.html')

@app.route('/styles.css')
def styles():
    return send_from_directory(STATIC_FOLDER, 'styles.css')

@app.route('/scripts/<path:path>')
def serve_script(path):
    return send_from_directory(os.path.join(STATIC_FOLDER, 'scripts'), path)

@app.route('/frontend/<path:filename>')
def serve_frontend(filename):
    return send_from_directory(STATIC_FOLDER, filename)

# === Live Monitoring Endpoint ===
@app.route('/live-data')
def live_data():
    ecg_pattern = [
        304.2, 312.3, 293.5, 362.4, 323.4, 301.7, 534.1,
        317.2, 309.3, 286.5, 375.4, 329.4, 311.7, 581.1
    ]
    return jsonify({
        "ecg": str(random.choice(ecg_pattern)),
        "hr": random.randint(60, 110),
        "spo2": round(random.uniform(94, 99), 1),
        "temp": round(random.uniform(36.5, 38.0), 1)
    })



# === Start App ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
