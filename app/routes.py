from flask import Blueprint, request, jsonify
import mysql.connector
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from .preprocess import preprocess_text
import json

main = Blueprint('main', __name__)

def connect_to_database():
    return mysql.connector.connect(
        host='your_host',
        user='your_user',
        password='your_password',
        database='your_database'
    )

tfidf_vectorizer = TfidfVectorizer(preprocessor=preprocess_text)

def vectorize_text(text):
    tfidf_matrix = tfidf_vectorizer.fit_transform([text])
    return tfidf_matrix.toarray()[0]

def get_usercity_vector(user_id, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT usercity_vector FROM user_tfidf_vectors WHERE user_id = %s", (user_id,))
    usercity_vector = cursor.fetchone()
    cursor.close()
    if usercity_vector:
        return np.array(json.loads(usercity_vector[0]))
    return None

def get_store_vectors_by_category(conn, category):
    cursor = conn.cursor()
    cursor.execute("SELECT store_id, store_vector FROM store_tfidf_vectors WHERE category = %s", (category,))
    store_vectors = cursor.fetchall()
    cursor.close()
    return store_vectors

@main.route('/recommendations', methods=['POST'])
def recommend():
    data = request.get_json()
    user_id = data.get('user_id', '')
    category = data.get('category', '')
    if user_id and category:
        conn = connect_to_database()
        usercity_vector = get_usercity_vector(user_id, conn)
        if usercity_vector is None:
            return jsonify({'error': 'User vector not found.'}), 404
        
        store_vectors = get_store_vectors_by_category(conn, category)
        store_recommendations = []
        for store_id, store_vector in store_vectors:
            store_vector = np.array(json.loads(store_vector))
            similarity_score = cosine_similarity([usercity_vector], [store_vector])[0][0]
            store_recommendations.append((store_id, similarity_score))
        
        store_recommendations.sort(key=lambda x: x[1], reverse=True)
        top_store_recommendations = store_recommendations[:10]
        
        conn.close()
        return jsonify({
            'user_id': user_id,
            'top_store_recommendations': top_store_recommendations,
        })
    else:
        return jsonify({'error': 'User ID and category are required.'}), 400

@main.route('/vectorize_usercity', methods=['POST'])
def vectorize_user():
    data = request.get_json()
    user_id = data.get('user_id', '')
    city = data.get('city', '')
    if user_id and city:
        usercity_vector = vectorize_text(city)
        return jsonify({'user_id': user_id, 'vector': usercity_vector.tolist()})
    else:
        return jsonify({'error': 'User ID and city are required.'}), 400

@main.route('/vectorize_store', methods=['POST'])
def vectorize_store():
    data = request.get_json()
    store_id = data.get('id', '')
    city = data.get('city', '')
    if store_id and city:
        store_vector = vectorize_text(city)
        return jsonify({'id': store_id, 'vector': store_vector.tolist()})
    else:
        return jsonify({'error': 'Store ID, city are required.'}), 400
