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
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECIPIENT
        msg['Subject'] = "🚨 EMERGENCY: Blood Clot Detected - Immediate Action Required"
        
        # HTML Email Body
        html_body = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Emergency Alert - Blood Clot Detected</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #1a0000 0%, #570000 100%);
                    color: #ffffff;
                }}
                .container {{
                    max-width: 600px;
                    margin: 0 auto;
                    background: rgba(43, 0, 0, 0.95);
                    border-radius: 20px;
                    overflow: hidden;
                    box-shadow: 0 8px 32px 0 rgba(255, 59, 59, 0.3);
                }}
                .header {{
                    background: linear-gradient(90deg, #ff3b3b, #8b0000);
                    padding: 30px;
                    text-align: center;
                    border-bottom: 3px solid #ff3b3b;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 28px;
                    font-weight: bold;
                    text-shadow: 0 0 10px #ff3b3b;
                    animation: pulse 2s infinite;
                }}
                @keyframes pulse {{
                    0% {{ text-shadow: 0 0 10px #ff3b3b; }}
                    50% {{ text-shadow: 0 0 20px #ff3b3b, 0 0 30px #ff3b3b; }}
                    100% {{ text-shadow: 0 0 10px #ff3b3b; }}
                }}
                .alert-icon {{
                    font-size: 48px;
                    margin-bottom: 15px;
                    display: block;
                }}
                .content {{
                    padding: 30px;
                }}
                .emergency-section {{
                    background: rgba(255, 59, 59, 0.1);
                    border-left: 5px solid #ff3b3b;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                }}
                .symptoms-section {{
                    background: rgba(255, 255, 255, 0.05);
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                    border: 1px solid rgba(255, 59, 59, 0.3);
                }}
                .location-section {{
                    background: rgba(255, 255, 255, 0.05);
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 10px;
                    border: 1px solid rgba(255, 59, 59, 0.3);
                }}
                .map-container {{
                    background: #2b0000;
                    padding: 15px;
                    border-radius: 10px;
                    margin: 15px 0;
                    text-align: center;
                }}
                .coordinates {{
                    background: rgba(255, 59, 59, 0.2);
                    padding: 10px;
                    border-radius: 8px;
                    margin: 10px 0;
                    font-family: 'Courier New', monospace;
                    font-weight: bold;
                }}
                .action-buttons {{
                    text-align: center;
                    margin: 30px 0;
                }}
                .btn {{
                    display: inline-block;
                    padding: 15px 30px;
                    margin: 10px;
                    background: linear-gradient(90deg, #ff3b3b, #8b0000);
                    color: white;
                    text-decoration: none;
                    border-radius: 25px;
                    font-weight: bold;
                    box-shadow: 0 4px 15px rgba(255, 59, 59, 0.3);
                    transition: all 0.3s ease;
                }}
                .btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(255, 59, 59, 0.5);
                }}
                .footer {{
                    background: rgba(0, 0, 0, 0.3);
                    padding: 20px;
                    text-align: center;
                    font-size: 12px;
                    color: #cccccc;
                }}
                .timestamp {{
                    background: rgba(255, 255, 255, 0.1);
                    padding: 10px;
                    border-radius: 8px;
                    margin: 15px 0;
                    text-align: center;
                    font-family: 'Courier New', monospace;
                }}
                .warning {{
                    color: #ff6b6b;
                    font-weight: bold;
                }}
                .info {{
                    color: #87ceeb;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <span class="alert-icon">🚨</span>
                    <h1>EMERGENCY ALERT</h1>
                    <p style="margin: 10px 0 0 0; font-size: 18px;">Blood Clot Detected - Immediate Action Required</p>
                </div>
                
                <div class="content">
                    <div class="emergency-section">
                        <h2 style="color: #ff3b3b; margin-top: 0;">🚨 CRITICAL SITUATION</h2>
                        <p><strong>ClotCare has detected a potential pulmonary embolism (blood clot in the lungs).</strong></p>
                        <p class="warning">This is a life-threatening emergency requiring immediate medical attention.</p>
                    </div>
                    
                    <div class="symptoms-section">
                        <h3 style="color: #ff3b3b; margin-top: 0;">⚠️ Common Symptoms Detected:</h3>
                        <ul style="color: #ffd6d6;">
                            <li><strong>Chest pain</strong> - Sharp, stabbing pain that worsens with breathing</li>
                            <li><strong>Shortness of breath</strong> - Difficulty breathing or rapid breathing</li>
                            <li><strong>Rapid heart rate</strong> - Increased pulse and palpitations</li>
                            <li><strong>Coughing</strong> - May include blood or bloody sputum</li>
                            <li><strong>Dizziness or fainting</strong> - Lightheadedness or loss of consciousness</li>
                            <li><strong>Leg swelling</strong> - Pain, warmth, or redness in legs</li>
                        </ul>
                    </div>
                    
                    <div class="location-section">
                        <h3 style="color: #ff3b3b; margin-top: 0;">📍 Patient Location</h3>
                        <div class="coordinates">
                            <strong>Latitude:</strong> 30.386<br>
                            <strong>Longitude:</strong> 30.489
                        </div>
                        <div class="map-container">
                            <p class="info">📍 <strong>Location:</strong> Qena Governorate, Egypt</p>
                            <p style="font-size: 14px; color: #cccccc;">Map coordinates provided for emergency response</p>
                        </div>
                    </div>
                    
                    <div class="timestamp">
                        <strong>Alert Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    </div>
                    
                    <div class="action-buttons">
                        <a href="tel:+201277294555" class="btn">📞 Call Emergency</a>
                        <a href="https://maps.google.com/?q=30.386,30.489" class="btn" target="_blank">🗺️ View on Map</a>
                    </div>
                    
                    <div style="background: rgba(255, 59, 59, 0.1); padding: 20px; border-radius: 10px; margin: 20px 0;">
                        <h3 style="color: #ff3b3b; margin-top: 0;">🚑 IMMEDIATE ACTIONS REQUIRED:</h3>
                        <ol style="color: #ffd6d6;">
                            <li><strong>Call emergency services immediately</strong> (911 or local emergency number)</li>
                            <li><strong>Do not move the patient</strong> unless in immediate danger</li>
                            <li><strong>Keep the patient calm</strong> and in a comfortable position</li>
                            <li><strong>Monitor vital signs</strong> - breathing, pulse, consciousness</li>
                            <li><strong>Prepare for rapid transport</strong> to the nearest hospital</li>
                        </ol>
                    </div>
                </div>
                
                <div class="footer">
                    <p><strong>ClotCare Emergency Response System</strong></p>
                    <p>Automated Alert - Powered by AI</p>
                    <p>This is an automated emergency notification. Please respond immediately.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text fallback
        text_body = f"""
        🚨 EMERGENCY ALERT - BLOOD CLOT DETECTED 🚨

        CRITICAL SITUATION: ClotCare has detected a potential pulmonary embolism (blood clot in the lungs).

        This is a life-threatening emergency requiring immediate medical attention.

        PATIENT LOCATION:
        Latitude: 30.386
        Longitude: 30.489
        Location: Qena Governorate, Egypt

        COMMON SYMPTOMS:
        - Chest pain (sharp, stabbing pain)
        - Shortness of breath
        - Rapid heart rate
        - Coughing (may include blood)
        - Dizziness or fainting
        - Leg swelling

        IMMEDIATE ACTIONS REQUIRED:
        1. Call emergency services immediately
        2. Do not move the patient unless in immediate danger
        3. Keep the patient calm
        4. Monitor vital signs
        5. Prepare for rapid transport to hospital

        Alert Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        ---
        ClotCare Emergency Response System
        Automated Alert - Powered by AI
        """
        
        # Attach both HTML and plain text versions
        msg.attach(MIMEText(text_body, 'plain'))
        msg.attach(MIMEText(html_body, 'html'))
        
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
