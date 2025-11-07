from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import threading
import json
from datetime import datetime
from sqlalchemy import create_engine, text
from producer.producer import connect_channel, send_payload
from .consumer_result import start_result_listener

app = Flask(__name__)
CORS(app)

# --- Configuración de base de datos ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/pokeguesser")
engine = create_engine(DATABASE_URL)

# --- Caché temporal para resultados ---
RESULT_CACHE = {}

# --- Endpoint para predicción ---
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Datos JSON requeridos"}), 400

    required_fields = ["types", "color", "height"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Faltan campos requeridos"}), 400

    payload = {
        "types": [t.strip().lower() for t in data["types"]],
        "color": data["color"].strip().lower(),
        "height": int(data["height"])
    }

    insert_sql = text("""
        INSERT INTO descripciones (texto, fecha_envio)
        VALUES (:texto, now())
        RETURNING id
    """)

    try:
        with engine.begin() as conn:
            r = conn.execute(insert_sql, {"texto": json.dumps(payload, ensure_ascii=False)})
            descripcion_id = r.fetchone()[0]
    except Exception as e:
        return jsonify({"error": f"DB insert error: {e}"}), 500

    try:
        connection, channel = connect_channel()
        envelope = {"descripcion_id": descripcion_id, "payload": payload}
        send_payload(channel, envelope)
        connection.close()
    except Exception as e:
        return jsonify({"error": f"RabbitMQ publish error: {e}"}), 500

    return jsonify({"descripcion_id": descripcion_id, "status": "queued"})


# --- Endpoint para consultar resultado ---
@app.route("/api/predict/result/<int:descripcion_id>", methods=["GET"])
def get_result(descripcion_id):
    result = RESULT_CACHE.get(descripcion_id)
    if not result:
        return jsonify({"status": "pending"})
    return jsonify({"status": "ready", "result": result})


# --- Endpoint para obtener descripción ---
@app.route("/api/description/<int:descripcion_id>", methods=["GET"])
def get_description(descripcion_id):
    q = text("SELECT id, texto, fecha_envio FROM descripciones WHERE id = :id")
    with engine.connect() as conn:
        row = conn.execute(q, {"id": descripcion_id}).fetchone()
    if not row:
        return jsonify({"error": "descripcion_id not found"}), 404
    return jsonify({
        "id": int(row.id),
        "texto": json.loads(row.texto),
        "fecha_envio": str(row.fecha_envio)
    })


# --- Iniciar el listener de resultados ---
def start_background_listener():
    listener_thread = threading.Thread(target=start_result_listener, args=(RESULT_CACHE,), daemon=True)
    listener_thread.start()


if __name__ == "__main__":
    start_background_listener()
    app.run(host="0.0.0.0", port=5000)
