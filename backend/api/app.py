from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import threading
import json
import traceback
from datetime import datetime
from sqlalchemy import create_engine, text
from producer.producer import connect_channel, send_payload
from .consumer_result import start_result_listener  # ✅ FIX IMPORT

app = Flask(__name__)
CORS(app)

# --- Configuración de base de datos ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/pokeguesser")
engine = create_engine(DATABASE_URL)

# --- Caché temporal para resultados ---
RESULT_CACHE = {}

# ================================================================
#                       ENDPOINT DE PREDICCIÓN
# ================================================================
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        print("⚠️  No se recibieron datos JSON.")
        return jsonify({"error": "Datos JSON requeridos"}), 400

    required_fields = ["types", "color", "height"]
    if not all(field in data for field in required_fields):
        print("⚠️  Faltan campos requeridos en el JSON:", data)
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
            print(f"✅ Inserción correcta en DB. descripcion_id={descripcion_id}")
    except Exception as e:
        print("❌ Error al insertar en la base de datos:", e)
        traceback.print_exc()
        return jsonify({"error": f"DB insert error: {str(e)}"}), 500

    try:
        connection, channel = connect_channel()
        envelope = {"descripcion_id": descripcion_id, "payload": payload}
        send_payload(channel, envelope)
        connection.close()
        print(f"✅ Mensaje publicado en RabbitMQ: {envelope}")
    except Exception as e:
        print("❌ Error al publicar en RabbitMQ:", e)
        traceback.print_exc()
        return jsonify({"error": f"RabbitMQ publish error: {str(e)}"}), 500

    return jsonify({"descripcion_id": descripcion_id, "status": "queued"})


# ================================================================
#                       GET RESULT
# ================================================================
@app.route("/api/predict/result/<int:descripcion_id>", methods=["GET"])
def get_result(descripcion_id):
    result = RESULT_CACHE.get(descripcion_id)
    if not result:
        return jsonify({"status": "pending"})
    return jsonify({"status": "ready", "result": result})


# ================================================================
#                       GET DESCRIPTION
# ================================================================
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


# ================================================================
#      THREAD DEL LISTENER (CON REINTENTOS AUTOMÁTICOS)
# ================================================================
def start_background_listener():
    print("🔁 Iniciando listener de resultados en background...")
    listener_thread = threading.Thread(target=start_result_listener, args=(RESULT_CACHE,), daemon=True)
    listener_thread.start()


if __name__ == "__main__":
    start_background_listener()
    print("🚀 Flask backend iniciado en modo debug")
    app.run(host="0.0.0.0", port=5000, debug=True)
