from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/login": {"origins": "http://localhost:8080"}})

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        port=os.getenv('DB_PORT')
    )

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute(
            'SELECT * FROM users WHERE username = %s AND password = %s',
            (username, password)
        )
        user = cur.fetchone()

        if user:
            return jsonify({'message': 'Login Successful'}), 200
        else:
            return jsonify({'message': 'Invalid Credentials'}), 401

    except Exception as e:
        return jsonify({'message': str(e)}), 500
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)