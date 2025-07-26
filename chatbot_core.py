import os
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pyttsx3
import threading
import uuid
import sys
import json
from Predict_db import get_latest_predictions
import speech_recognition as sr


# Load environment variables
load_dotenv()

class ChatBotCore:
    def __init__(self):
        self.groq_api_key = os.environ.get('GROQ_API_KEY')
        if not self.groq_api_key:
            print("GROQ_API_KEY not found in environment variables")
            sys.exit(1)
     
        self.chat_history = []
        self.current_conversation = None
        self.session_id = str(uuid.uuid4())
        self.history_file = f"chat_history.json"
        
        # Track active TTS for cancellatiaon
        self.active_tts_thread = None
        self.is_speaking = False
        self.last_spoken_text = ""
        self.is_stopped_manually = False
        
        # Initialize text-to-speech engine
        try:
            self.engine = pyttsx3.init()
            print("TTS engine initialized")
        except Exception as e:
            print(f"Error initializing pyttsx3: {e}")
            self.engine = None
        
        # TTS settings
        self.always_speak = True
        
        # Load existing history
        self.load_history()
    
    def save_history(self):
        """Save chat history to JSON file"""
        with open(self.history_file, "w", encoding="utf-8") as f:
         json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
    def load_history(self):
        """Load chat history from JSON file"""
        try:
           with open(self.history_file, "r", encoding="utf-8") as f:
            self.chat_history = json.load(f)
        except FileNotFoundError:
            self.chat_history = []
    
    def text_to_speech(self, text):
        """Speaks the given text using pyttsx3."""
        try:
            self.stop_speaking()

            engine = pyttsx3.init()
            voices = engine.getProperty('voices')

            for voice in voices:
                if "hindi" in voice.name.lower() or "marathi" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break

            engine.say(text)
            engine.runAndWait()
            engine.stop()

        except Exception as e:
            print(f"TTS error: {e}")

        
    
    def stop_speaking(self):
        try:
            if hasattr(self, 'engine') and self.engine:
                self.engine.stop()
        except Exception as e:
            print(f"Stop Speaking Error: {e}")


    def initialize_conversation(self, model="llama3-8b-8192", memory_length=5):
        """Initialize or reset the conversation chain"""
        groq_chat = ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model
        )
        memory = ConversationBufferWindowMemory(k=memory_length, return_messages=True)
        self.current_conversation = ConversationChain(
            llm=groq_chat,
            memory=memory
        )
        
        # Restore previous conversation context if history exists
        if self.chat_history:
            for chat in self.chat_history[-memory_length:]:
                if not chat.get('error'):
                    memory.save_context({"input": chat['human']}, {"output": chat['AI']})
        
        return self.current_conversation
    
    def process_message(self, user_input, model="llama3-8b-8192", memory_length=5, speak=None):
        """Process user input and return appropriate response"""
        if not self.current_conversation:
            self.initialize_conversation(model, memory_length)
            
        try:
            # Stop any ongoing speech
            self.stop_speaking()
            
            # Define clear agricultural advice triggers
            advice_triggers = [
                'advice', 'suggest', 'recommend', 'help with farming',
                'what should i do about', 'how to treat', 'crop suggestion',
                'fertilizer advice', 'disease treatment'
            ]
            
            # Check if this is an agricultural advice request
            is_agri_advice = any(
                trigger in user_input.lower() 
                for trigger in advice_triggers
            ) and any(
                kw in user_input.lower()
                for kw in ['crop', 'plant', 'soil', 'disease', 'fertilizer', 'farm']
            )
            
            if is_agri_advice:
                # Get agricultural context if available
                context = self.get_context_predictions()
                if context:
                    prompt = (
                        f"User is asking for agricultural advice. "
                        f"Provide specific recommendations ONLY if relevant to this context:\n\n"
                        f"{context}\n\n"
                        f"Format response as:\n"
                        f"🌱 CROP ADVICE: [if crop data exists]\n"
                        f"🦠 DISEASE HELP: [if disease detected]\n"
                        f"🧪 FERTILIZER TIPS: [if fertilizer data exists]\n"
                        f"Only include sections that have supporting data."
                    )
                    user_input = prompt
                else:
                    user_input = (
                        "I need agricultural advice but haven't used prediction tools yet. "
                        "Ask me to first provide soil/crop/disease details or use the prediction tools."
                    )
            else:
                # Regular conversation - don't modify the input
                pass
                
            # Get response from LLM
            response = self.current_conversation(user_input)
            text_response = response['response']
            
            # Save to history
            self.chat_history.append({
                'human': user_input,
                'AI': text_response,
                'was_advice': is_agri_advice
            })
            self.save_history()
            
            # Handle TTS if enabled
            if speak if speak is not None else self.always_speak:
                threading.Timer(0.1, lambda: self.text_to_speech(text_response)).start()
                
            return text_response
            
        except Exception as e:
            error_msg = f"Error: {str(e)}. Please try again."
            self.chat_history.append({
                'human': user_input,
                'AI': error_msg,
                'error': True
            })
            self.save_history()
            return error_msg
    


    def speech_to_text(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = r.listen(source, timeout=5)
        try:
            text = r.recognize_google(audio, language="hi-IN")  # or "mr-IN" for Marathi
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            print(f"STT Error: {e}")
            return ""


    
    def reset_session(self):
        """Reset the session, stop any TTS and clear history"""
        # Stop any ongoing speech
        self.stop_speaking()
        
        # Clear chat history and save
        self.chat_history = []
        self.save_history()
        
        # Reset conversation
        self.initialize_conversation()
        
        return {
            "status": "success",
            "message": "Session reset successfully",
            "session_id": self.session_id
        }
    def get_context_predictions(self):
        """
        Get the latest predictions from all models and format them for LLM context
        Returns:
            str: Formatted prediction context or empty string if no predictions
        """
        predictions = get_latest_predictions()
        context_parts = []
        
        # Format crop prediction if available
        if predictions['crop_prediction']:
            crop = predictions['crop_prediction']
            context_parts.append(
                f"🌱 Latest Crop Recommendation:\n"
                f"- Crop: {crop['recommended_crop']}\n"
                f"- Soil Nutrients: N:{crop['N']}, P:{crop['P']}, K:{crop['K']}\n"
                f"- Conditions: Temp:{crop['temperature']}°C, Humidity:{crop['humidity']}%, "
                f"Rainfall:{crop['rainfall']}mm, pH:{crop['ph']}"
            )
        
        # Format disease prediction if available
        if predictions['disease_prediction']:
            disease = predictions['disease_prediction']
            context_parts.append(
                f"🦠 Latest Disease Detection:\n"
                f"- Prediction: {disease['prediction']} "
                f"(Confidence: {float(disease['confidence']):.1f}%)\n"
                f"{'- Image: ' + disease['image_path'] if disease.get('image_path') else ''}"
            )
        
        # Format fertilizer prediction if available
        if predictions['fertilizer_prediction']:
            fert = predictions['fertilizer_prediction']
            context_parts.append(
                f"🧪 Latest Fertilizer Recommendation:\n"
                f"- Fertilizer: {fert['recommended_fertilizer']}\n"
                f"- For Crop: {fert['crop_type']}\n"
                f"- Soil Type: {fert['soil_type']}\n"
                f"- Nutrients: N:{fert['nitrogen']}, P:{fert['phosphorous']}, K:{fert['potassium']}"
            )
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    