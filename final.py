from flask import Flask, request, jsonify, send_from_directory, abort, session
from flask_cors import CORS
import pickle
import numpy as np
import os
from PIL import Image
import logging
from io import BytesIO
import joblib
from chatbot_core import ChatBotCore
from dotenv import load_dotenv
import uuid
from Predict_db import save_disease_prediction, save_crop_recommendation
load_dotenv()
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
import simpleaudio as sa

def speak(text):
    tts = gTTS(text=text, lang='hi')  # or 'en' if needed
    tts.save("response.mp3")
    audio = AudioSegment.from_file("response.mp3", format="mp3")
    play_obj = sa.play_buffer(audio.raw_data, audio.channels, audio.sample_width, audio.frame_rate)
    play_obj.wait_done()

def voice_chat_loop():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        try:
            with mic as source:
                print("🎤 Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)

            user_input = recognizer.recognize_google(audio, language="hi-IN")  # or auto-detect
            print("🗣️ You said:", user_input)

            bot = get_chatbot()
            response = bot.ask(user_input)
            print("🤖 Bot:", response)

            speak(response)

        except sr.UnknownValueError:
            print("🤔 Could not understand audio")
        except Exception as e:
            print("❌ Error:", e)
            break


# Import Keras functions for model loading and image processing
from keras.models import load_model


app = Flask(__name__, static_folder='static')
CORS(app)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', os.urandom(24))

# --- 1. LOAD THE NEW DISEASE DETECTION MODEL (WITH FIX) ---
DISEASE_MODEL_PATH = "mobilenetv2_crop_model2.h5"
disease_model = load_model(DISEASE_MODEL_PATH, compile=False) # <--- FIX APPLIED HERE

# --- 2. UPDATE CLASS NAMES AND IMAGE SIZE ---
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Tomato__Target_Spot',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_healthy',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite'
]

# --- Chatbot Initialization (Unchanged) ---
chatbots = {}
def get_chatbot():
    """Get or create a chatbot instance for the current session"""
    # Get or create a session ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        
    session_id = session['session_id']
    
    # Create a new bot if needed
    if session_id not in chatbots:
        chatbots[session_id] = ChatBotCore()
    
    return chatbots[session_id]

# --- App Configuration and Model Loading (Unchanged for other features) ---
CONFIDENCE_THRESHOLD = 0.7
MAX_FILE_SIZE = 16 * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Note on scikit-learn warnings:
# The "InconsistentVersionWarning" from scikit-learn is not a fatal error.
# It means your .pkl files were created with a different version of scikit-learn.
# The code will likely still work, but for best practice, you should re-save your
# pickle files (model.pkl, fertilizer_model.pkl, etc.) using the same version
# of scikit-learn you are running now (version 1.7.0).
crop_model = pickle.load(open('model.pkl', 'rb'))
fert_model = joblib.load('fertilizer_model.pkl')
soil_encoder = joblib.load('soil_encoder.pkl')
crop_encoder = joblib.load('crop_encoder.pkl')
fertilizer_encoder = joblib.load('fertilizer_encoder.pkl')
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# --- Error Handling (Unchanged) ---
@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(500)
@app.errorhandler(413)
def handle_error(error):
    code = getattr(error, 'code', 500)
    description = getattr(error, 'description', str(error))
    response = jsonify({"error": description, "status": "error", "code": code})
    response.status_code = code
    return response

# --- Image Preprocessing (Updated to use IMG_SIZE) ---
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize(IMG_SIZE) # Use the model's expected size
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        abort(400, description=f"Invalid image: {str(e)}")

# --- Static Routes (Unchanged) ---
@app.route('/')
@app.route('/home')
def home():
    return send_from_directory('static', 'landingpage.html')

# --- PREDICTION ROUTES ---

@app.route('/predict-crop', methods=['POST'])
def predict_crop():
    # This function is unchanged
    try:
        data = request.get_json()
        features = [
            data['N'], data['P'], data['K'], data['temperature'],
            data['humidity'], data['ph'], data['rainfall']
        ]
        final_input = np.array([features])
        prediction = crop_model.predict(final_input)[0]
        session_id = session.get('session_id', str(uuid.uuid4()))
        save_crop_recommendation(
            session_id=session_id, recommended_crop=prediction, **data
        )
        return jsonify({'recommended_crop': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict-fertilizer', methods=['POST'])
def predict_fertilizer():
    # This function is unchanged
    try:
        data = request.get_json(force=True)
        # ... (rest of the function is identical)
        app.logger.info(f"Received data: {data}")

        required_fields = ['Temperature', 'Humidity', 'Moisture',
                           'Soil Type', 'Crop Type', 'Nitrogen',
                           'Phosphorous', 'Potassium']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        soil_input = data['Soil Type'].strip().capitalize()
        crop_input = data['Crop Type'].strip().lower()

        if soil_input not in soil_encoder.classes_:
            return jsonify({'error': f"Invalid soil type '{soil_input}'. Available types: {list(soil_encoder.classes_)}"}), 400
        if crop_input not in crop_encoder.classes_:
            return jsonify({'error': f"Invalid crop type '{crop_input}'. Available types: {list(crop_encoder.classes_)}"}), 400

        soil_encoded = soil_encoder.transform([soil_input])[0]
        crop_encoded = crop_encoder.transform([crop_input])[0]

        features = np.array([[data['Temperature'], data['Humidity'], data['Moisture'],
                              soil_encoded, crop_encoded,
                              data['Nitrogen'], data['Phosphorous'], data['Potassium']]])

        prediction_encoded = fert_model.predict(features)[0]
        fertilizer_name = fertilizer_encoder.inverse_transform([prediction_encoded])[0]

        app.logger.info(f"Predicted Fertilizer: {fertilizer_name}")
        return jsonify({'recommended_fertilizer': fertilizer_name})

    except ValueError as ve:
        app.logger.error(f"Encoding error: {str(ve)}")
        return jsonify({'error': f'Encoding error: {str(ve)}'}), 400

    except Exception as e:
        app.logger.error(f"Fertilizer prediction failed: {str(e)}")
        return jsonify({'error': f'Fertilizer prediction failed: {str(e)}'}), 500
        
# --- UPDATED /predict ENDPOINT FOR DISEASE DETECTION ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        abort(400, description="No file part in the request")
    
    file = request.files['file']
    
    if file.filename == '':
        abort(400, description="No file selected")
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        abort(400, description="Only PNG/JPG/JPEG images are allowed")
    
    try:
        # Step 1: Read and preprocess image
        image_bytes = file.read()
        image_array = preprocess_image(image_bytes)
        image_batch = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Step 2: Perform prediction directly with the loaded model
        predictions = disease_model.predict(image_batch)
        predicted_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]

        # Step 3: Prepare response
        response_data = {
            "prediction": predicted_class,
            "confidence": round(float(confidence) * 100, 2)
        }
        
        if confidence < CONFIDENCE_THRESHOLD:
            response_data.update({
                "status": "low_confidence",
                "message": "Model isn't confident about this prediction"
            })
        else:
            response_data["status"] = "success"
            # Optional: Store prediction in the database
            try:
                session_id = session.get('session_id', str(uuid.uuid4()))
                save_disease_prediction(
                    session_id=session_id,
                    prediction=predicted_class,
                    confidence=response_data["confidence"],
                    image_path=file.filename if file else None
                )
            except Exception as db_error:
                logger.error(f"Database insertion failed: {str(db_error)}")
                response_data["database_status"] = "failed"
                response_data["database_error"] = str(db_error)

        return jsonify(response_data)

    except Exception as general_error:
        logger.error(f"Prediction failed: {str(general_error)}")
        abort(500, description=str(general_error))


# --- AI Assistant and Chatbot Routes (Unchanged) ---
@app.route('/api/chat', methods=['POST'])
def chat():
    """Process a text chat message"""
    chatbot = get_chatbot()
    data = request.json
    print("Incoming data:", data)
    user_input = data.get('message', '')
    model = data.get('model', 'llama3-8b-8192')
    memory_length = data.get('memory_length', 5)
    use_tts = data.get('use_tts', True)
    
    prompt_mode = data.get('prompt_mode', 'farmer')

    if prompt_mode == 'farmer':
        system_prompt = (
            "You are KrushiGPT, an AI assistant designed to help Indian farmers. "
                "Provide accurate, simple, and practical advice only on agriculture, fertilizers, crop diseases, "
                "and weather in Marathi or simple English.\n\n"
               "STRICT RESPONSE RULES:"
                "1. Each bullet on NEW LINE with BLANK LINE before it"
                "2. Use VARIABLE EMOJIS based on content:"
                    "- Crops: 🌾(wheat), 🌱(seedling), 🍚(rice), 🥜(groundnut)"
                    "- Actions: 💧(water), ✂️(prune), 🌿(organic), 🧪(chemical)"
                    "- Problems: 🐛(pest), 🦠(disease), ⚠️(warning), 🔥(blight)"
                    "- Weather: ☀️(sun), 🌧️(rain), 🌪️(storm), ❄️(frost)"
                "3. Never repeat same emoji consecutively"
                "4. Match emoji to Marathi/English content"
        )
    else:
        system_prompt = (
            "You are a general-purpose helpful assistant. Be friendly and informative."
        )

    full_prompt = f"{system_prompt}\nUser: {user_input}"

    # Process the message
    response_text = chatbot.process_message(full_prompt, model, memory_length, speak=use_tts)
    
    result = {
        'response': response_text,
        'chatHistory': chatbot.chat_history,  # Return chat history
        'speech_enabled': use_tts
    }
    
    return jsonify(result)

@app.route('/api/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech using the chatbot's TTS engine"""
    chatbot = get_chatbot()
    data = request.json
    text = data.get('text', '')
    
    # Use the chatbot's TTS function
    chatbot.text_to_speech(text)
    
    return jsonify({
        'status': 'success',
        'message': 'TTS processing started'
    })

@app.route('/api/stop-tts', methods=['POST'])
def stop_tts():
    """Stop ongoing TTS playback"""
    chatbot = get_chatbot()
    # Ensure speech is stopped immediately
    chatbot.stop_speaking()
    
    return jsonify({
        'status': 'success',
        'message': 'Speech stopped'
    })

@app.route('/api/voice', methods=['POST'])
def voice_input():
    """Process voice input from the frontend"""
    chatbot = get_chatbot()
    data = request.json
    text = data.get('text', '')
    model = data.get('model', 'llama3-8b-8192')
    memory_length = data.get('memory_length', 5)
    use_tts_response = data.get('use_tts_response', True)
    
    prompt_mode = data.get('prompt_mode', 'farmer')

    if prompt_mode == 'farmer':
        system_prompt = (
            "You are **KrushiGPT**, an AI assistant designed to help Indian farmers.\n\n"
            "🧠 Your job: Provide accurate, simple, and practical advice only on:\n"
            "- Agriculture\n"
            "- Fertilizers\n"
            "- Crop diseases\n"
            "- Weather\n"
            "Reply in **Marathi** or **simple English** depending on the user input.\n\n"
            
            "📌 **STRICT RESPONSE RULES**:\n\n"
            
            "1️⃣ Each bullet point should be on a **new line**, with a **blank line before it**.\n\n"
            
            "2️⃣ Use **appropriate emojis** based on content:\n"
            "- 🌾 (wheat), 🌱 (seedling), 🍚 (rice), 🥜 (groundnut)\n"
            "- 💧 (water), ✂️ (prune), 🌿 (organic), 🧪 (chemical)\n"
            "- 🐛 (pest), 🦠 (disease), ⚠️ (warning), 🔥 (blight)\n"
            "- ☀️ (sun), 🌧️ (rain), 🌪️ (storm), ❄️ (frost)\n\n"
            
            "3️⃣ **Never repeat the same emoji consecutively.**\n\n"
            
            "4️⃣ Match the emoji to the meaning of the content (English or Marathi).\n\n"
            
            "✅ Always be concise, direct, and helpful. Avoid unnecessary text."

            "Example:\n"
            "🌱 **Crop:** Tomato\n\n"
            "🦠 **Problem:** Early blight with brown leaf spots\n\n"
            "🧪 **Treatment:** Use Mancozeb (2 gm/litre) and spray every 7 days\n\n"
            "⚠️ **Tip:** Remove infected leaves to prevent spread"

        )

    else:
        system_prompt = (
            "You are a general-purpose helpful assistant. Be friendly and informative."
        )

    full_prompt = f"{system_prompt}\nUser: {text}"

    # Process the message
    response_text = chatbot.process_message(text, model, memory_length, speak=use_tts_response)
    
    result = {
        'text': text,
        'response': response_text,
        'speech_enabled': use_tts_response
    }
    
    return jsonify(result)


@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update chatbot settings (TTS preferences)"""
    chatbot = get_chatbot()
    data = request.json
    always_speak = data.get('always_speak', None)
    
    if always_speak is not None:
        chatbot.always_speak = always_speak
    
    return jsonify({
        'status': 'success',
        'settings': {
            'always_speak': chatbot.always_speak
        }
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the conversation and stop any ongoing speech"""
    chatbot = get_chatbot()
    data = request.json
    model = data.get('model', 'llama3-8b-8192')
    memory_length = data.get('memory_length', 5)
    
    # Reset the conversation (this will also stop any ongoing speech)
    chatbot.initialize_conversation(model, memory_length)
    
    return jsonify({
        'status': 'success',
        'message': 'Conversation reset successfully'
    })

@app.route('/api/listen', methods=['POST'])
def listen():
    """Activate speech recognition and return transcribed text"""
    chatbot = get_chatbot()
    
    # Use the chatbot's speech recognition function
    transcribed_text = chatbot.speech_to_text()
    
    if not transcribed_text:
        return jsonify({
            'status': 'error',
            'message': 'No speech detected or could not transcribe audio'
        }), 400
    
    return jsonify({
        'status': 'success',
        'text': transcribed_text
    })


@app.route('/api/stop-speaking', methods=['POST'])
def stop_speaking_endpoint():
    """Stop the chatbot from speaking without resetting the conversation"""
    chatbot = get_chatbot()
    # Stop the speech immediately
    chatbot.stop_speaking()
    
    return jsonify({
        "status": "success",
        "message": "Speech stopped successfully"
    })

@app.route('/DiseasePrediction')
def serve_detection():
    # This function is unchanged
    return send_from_directory('static', 'DiseasePrediction.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    # This function is unchanged
    return send_from_directory(app.static_folder, filename)

@app.route('/crop_recommendation')
def serve_crop_recommendation():
    # This function is unchanged
    return send_from_directory('static', 'crop_recommendation.html')

@app.route('/fertilizer_recommendation2')
def serve_fertilizer_recommendation():
    # This function is unchanged
    return send_from_directory('static', 'fertilizer_recommendation2.html')

@app.route('/forecast')
def serve_weather():
    # This function is unchanged
    return send_from_directory('static', 'forecast.html')

@app.route('/subsidy')
def serve_subsidy_finder():
    # This function is unchanged
    return send_from_directory('static', 'subsidy.html')

@app.route('/ai_assistant')
def serve_ai():
    # This function is unchanged
    return send_from_directory('static', 'ai_assistant.html')

if __name__ == '__main__':
    app.run(host='localhost', port=2025, debug=True)