import mysql.connector
import os
from dotenv import load_dotenv
from mysql.connector import pooling
import logging
from sentence_transformers import SentenceTransformer
import json 


load_dotenv()
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text: str):
    """Generate embedding using SentenceTransformer"""
    return embedder.encode(text)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Connection pool configuration
dbconfig = {
    "host": "localhost",
    "user": "root",
    "password": os.environ.get('SQL_PASSWORD'),
    "database": "nhr"
}

try:
    connection_pool = pooling.MySQLConnectionPool(
        pool_name="asep_pool",
        pool_size=5,
        **dbconfig
    )
    logger.info("Created connection pool with 5 connections")
except Exception as e:
    logger.error(f"Error creating connection pool: {str(e)}")
    connection_pool = None

def get_connection():
    """Get a connection from the pool"""
    if connection_pool:
        return connection_pool.get_connection()
    return mysql.connector.connect(**dbconfig)


def generate_embedding(text: str):
    """Generate embedding using SentenceTransformer"""
    return embedder.encode(text)


def save_disease_prediction(session_id, prediction, confidence, image_path=None):
    conn = None
    cursor = None
    embedding = generate_embedding(prediction)
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
        INSERT INTO disease_predictions 
        (session_id, prediction, confidence, image_path, embedding_vector) 
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            session_id, 
            prediction, 
            confidence, 
            image_path,
            json.dumps(embedding.tolist())  # Store as JSON
        ))
        conn.commit()
        logger.info(f"Disease prediction saved for session {session_id}")
    except Exception as e:
        logger.error(f"Error saving disease prediction: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def save_crop_recommendation(session_id, recommended_crop, N, P, K, temperature, humidity, ph, rainfall):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
        INSERT INTO crop_recommendations 
        (session_id, recommended_crop, N, P, K, temperature, humidity, ph, rainfall) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (session_id, recommended_crop, N, P, K, temperature, humidity, ph, rainfall))
        conn.commit()
        logger.info(f"Crop recommendation saved for session {session_id}")
    except Exception as e:
        logger.error(f"Error saving crop recommendation: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def save_fertilizer_recommendation(session_id, recommended_fertilizer, temperature, humidity, moisture, 
                                  nitrogen, potassium, phosphorous, soil_type, crop_type):
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        query = """
        INSERT INTO fertilizer_recommendations 
        (session_id, recommended_fertilizer, temperature, humidity, moisture, 
         nitrogen, potassium, phosphorous, soil_type, crop_type) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (session_id, recommended_fertilizer, temperature, humidity, moisture,
                              nitrogen, potassium, phosphorous, soil_type, crop_type))
        conn.commit()
        logger.info(f"Fertilizer recommendation saved for session {session_id}")
    except Exception as e:
        logger.error(f"Error saving fertilizer recommendation: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def get_latest_predictions():
    """
    Retrieve the latest prediction from each model (crop, disease, fertilizer)
    Returns:
        dict: {
            'crop_prediction': {...},
            'disease_prediction': {...},
            'fertilizer_prediction': {...}
        }
        Each prediction will be a single dictionary (not a list) or None if no prediction exists
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor(dictionary=True)
        
        result = {
            'crop_prediction': None,
            'disease_prediction': None,
            'fertilizer_prediction': None
        }
        
        # Get latest crop prediction
        cursor.execute("""
            SELECT * FROM crop_recommendations 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        crop_data = cursor.fetchone()
        if crop_data:
            result['crop_prediction'] = crop_data
        
        # Get latest disease prediction
        cursor.execute("""
            SELECT * FROM disease_predictions 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        disease_data = cursor.fetchone()
        if disease_data:
            result['disease_prediction'] = disease_data
        
        # Get latest fertilizer prediction
        cursor.execute("""
            SELECT * FROM fertilizer_recommendations 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        fertilizer_data = cursor.fetchone()
        if fertilizer_data:
            result['fertilizer_prediction'] = fertilizer_data
        
        logger.info("Successfully retrieved latest predictions")
        return result
        
    except Exception as e:
        logger.error(f"Error retrieving latest predictions: {str(e)}")
        return {
            'crop_prediction': None,
            'disease_prediction': None,
            'fertilizer_prediction': None
        }
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def print_latest_predictions():
    """
    Retrieve and print the latest predictions in a readable format
    """
    predictions = get_latest_predictions()
    
    print("\n=== Latest Predictions ===")
    
    # Print crop prediction
    crop = predictions['crop_prediction']
    if crop:
        print(f"\nCrop Recommendation:")
        print(f"- Recommended Crop: {crop['recommended_crop']}")
        print(f"- Soil Nutrients: N:{crop['N']}, P:{crop['P']}, K:{crop['K']}")
        print(f"- Weather Conditions: Temp:{crop['temperature']}°C, Humidity:{crop['humidity']}%, Rainfall:{crop['rainfall']}mm")
        print(f"- Soil pH: {crop['ph']}")
        print(f"- Timestamp: {crop['timestamp']}")
    else:
        print("\nNo crop recommendations found")
    
    # Print disease prediction
    disease = predictions['disease_prediction']
    if disease:
        print(f"\nDisease Prediction:")
        print(f"- Prediction: {disease['prediction']}")
        print(f"- Confidence: {float(disease['confidence']) * 100:.1f}%")
        if disease['image_path']:
            print(f"- Image: {disease['image_path']}")
        print(f"- Timestamp: {disease['timestamp']}")
    else:
        print("\nNo disease predictions found")
    
    # Print fertilizer prediction
    fertilizer = predictions['fertilizer_prediction']
    if fertilizer:
        print(f"\nFertilizer Recommendation:")
        print(f"- Recommended Fertilizer: {fertilizer['recommended_fertilizer']}")
        print(f"- For Crop: {fertilizer['crop_type']}")
        print(f"- Soil Type: {fertilizer['soil_type']}")
        print(f"- Nutrients: N:{fertilizer['nitrogen']}, P:{fertilizer['phosphorous']}, K:{fertilizer['potassium']}")
        print(f"- Conditions: Temp:{fertilizer['temperature']}°C, Humidity:{fertilizer['humidity']}%, Moisture:{fertilizer['moisture']}")
        print(f"- Timestamp: {fertilizer['timestamp']}")
    else:
        print("\nNo fertilizer recommendations found")