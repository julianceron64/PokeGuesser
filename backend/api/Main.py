# backend/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import json
import os
import uuid
from datetime import datetime
from producer.producer import connect_channel, send_payload  # reusar producer helper
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/pokeguesser")
engine = create_engine(DATABASE_URL)

app = FastAPI()

class PredictData(BaseModel):
    types: conlist(str, min_items=1, max_items=2)
    color: str
    height: int

@app.post("/api/predict")
def predict(data: PredictData):
    # Normalize payload
    payload = {
        "types": [t.strip().lower() for t in data.types],
        "color": data.color.strip().lower(),
        "height": int(data.height)
    }

    # Insert into descripciones (we store JSON string in texto)
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
        raise HTTPException(status_code=500, detail=f"DB insert error: {e}")

    # Publish envelope to RabbitMQ
    try:
        connection, channel = connect_channel()
        envelope = {
            "descripcion_id": descripcion_id,
            "payload": payload
        }
        send_payload(channel, envelope)
        connection.close()
    except Exception as e:
        # On failure update DB? We keep record in descripciones but inform error
        raise HTTPException(status_code=500, detail=f"RabbitMQ publish error: {e}")

    return {"descripcion_id": descripcion_id, "status": "queued"}

@app.get("/api/description/{descripcion_id}")
def get_description(descripcion_id: int):
    q = text("SELECT id, texto, fecha_envio FROM descripciones WHERE id = :id")
    with engine.connect() as conn:
        row = conn.execute(q, {"id": descripcion_id}).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="descripcion_id not found")
    return {"id": int(row.id), "texto": json.loads(row.texto), "fecha_envio": str(row.fecha_envio)}
