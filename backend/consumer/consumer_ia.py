# backend/consumer/consumer_math.py
#!/usr/bin/env python3
import os
import json
import time
import pika
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# envs
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/pokeguesser")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
QUEUE_IN = "pokemon_descriptions"
QUEUE_OUT = "pokemon_predictions"
EMBEDDINGS_PATH = os.path.join("db", "pokemon_embeddings.npy")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

engine = create_engine(DATABASE_URL)

# Load pokemons once
with engine.connect() as conn:
    pokemons = conn.execute(text("SELECT id, name, type, color, habitat, height, weight, description FROM pokemons")).fetchall()
print(f"Se cargaron {len(pokemons)} pokemons desde DB")

def parse_db_types(type_field: str):
    if not type_field:
        return []
    return [t.strip().lower() for t in type_field.split(",") if t.strip()]

def compute_type_score(user_types, db_types):
    if not user_types:
        return 0
    matches = sum(1 for t in user_types if t in db_types)
    if len(user_types) >= 2:
        return 2 if matches == len(user_types) else (1 if matches >= 1 else 0)
    else:
        return 1 if matches >= 1 else 0

def height_similarity(user_h, db_h):
    return 1.0 / (1.0 + abs(int(user_h) - int(db_h)))

def build_text_for_embedding(p):
    parts = [p.name or "", p.description or "", p.type or "", p.color or "", p.habitat or ""]
    return " ".join([str(x) for x in parts if x])

pokemon_texts = [build_text_for_embedding(p) for p in pokemons]

# embeddings lazy load
use_embeddings = False
if os.path.exists(EMBEDDINGS_PATH):
    try:
        embeddings = np.load(EMBEDDINGS_PATH)
        use_embeddings = True
        print("Embeddings cargados desde archivo para fallback")
    except Exception as e:
        print("Error cargando embeddings:", e)
        use_embeddings = False

model = SentenceTransformer(EMBEDDING_MODEL)  # model available for fallback

# Rabbit setup
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
in_channel = connection.channel()
out_channel = connection.channel()
in_channel.queue_declare(queue=QUEUE_IN, durable=True)
out_channel.queue_declare(queue=QUEUE_OUT, durable=True)

def publish_result_to_queue(out_channel, descripcion_id, payload, result):
    envelope = {
        "descripcion_id": descripcion_id,
        "payload": payload,
        "result": result
    }
    out_channel.basic_publish(
        exchange="",
        routing_key=QUEUE_OUT,
        body=json.dumps(envelope, ensure_ascii=False),
        properties=pika.BasicProperties(delivery_mode=2)
    )

def insert_intento(descripcion_id, numero_intento, pokemon_predicho, probabilidad, tiempo_ms):
    insert_sql = text("""
        INSERT INTO intentos
        (descripcion_id, numero_intento, pokemon_predicho, es_correcto, probabilidad, tiempo_respuesta_ms, fecha_intento)
        VALUES (:desc_id, :num, :poke, FALSE, :prob, :tms, now())
        RETURNING id
    """)
    with engine.begin() as conn:
        r = conn.execute(insert_sql, {"desc_id": descripcion_id, "num": numero_intento,
                                      "poke": pokemon_predicho, "prob": probabilidad, "tms": int(tiempo_ms)})
        return r.fetchone()[0]

def get_next_intento_number(descripcion_id):
    q = text("SELECT COUNT(*) FROM intentos WHERE descripcion_id = :desc_id")
    with engine.connect() as conn:
        cnt = conn.execute(q, {"desc_id": descripcion_id}).scalar()
    return int(cnt) + 1

def predict_structured(user_types, user_color, user_height, top_k=3):
    user_color = (user_color or "").strip().lower()
    user_types = [t.lower() for t in user_types] if user_types else []

    # Stage 1: exact color filter
    candidates = [p for p in pokemons if (p.color or "").strip().lower() == user_color]
    if candidates:
        scored = []
        for p in candidates:
            db_types = parse_db_types(p.type)
            tscore = compute_type_score(user_types, db_types)
            hsim = height_similarity(user_height, p.height or 0)
            score = tscore + hsim
            scored.append((score, p, tscore, hsim))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, p, tscore, hsim in scored[:top_k]:
            results.append({
                "id": p.id,
                "name": p.name,
                "score": float(score),
                "type_score": int(tscore),
                "height_similarity": float(hsim),
                "db_types": parse_db_types(p.type),
                "db_color": p.color,
                "db_height": p.height
            })
        return {"mode": "structured_color_filter", "results": results}

    # fallback to embeddings
    print("No hay candidatos por color exacto -> fallback embeddings")
    global use_embeddings, embeddings
    if not use_embeddings:
        embeddings = model.encode(pokemon_texts, show_progress_bar=True, convert_to_numpy=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        use_embeddings = True

    user_text = " ".join(user_types + [user_color, f"{user_height}dm"])
    user_emb = model.encode([user_text], convert_to_numpy=True)
    sims = cosine_similarity(user_emb, embeddings)[0]
    idx_sorted = sims.argsort()[::-1][:top_k]
    results = []
    for i in idx_sorted:
        p = pokemons[i]
        results.append({
            "id": p.id,
            "name": p.name,
            "score": float(sims[i]),
            "db_types": parse_db_types(p.type),
            "db_color": p.color,
            "db_height": p.height
        })
    return {"mode": "fallback_embeddings", "results": results}

def on_message(ch, method, properties, body):
    try:
        envelope = json.loads(body)
        descripcion_id = envelope.get("descripcion_id")
        payload = envelope.get("payload", {})
    except Exception as e:
        print("Mensaje inválido:", e)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    if descripcion_id is None:
        print("Aviso: descripcion_id es None — el mensaje no fue creado por la API. Se procesará igualmente con descripcion_id NULL.")
    print("Procesando descripcion_id:", descripcion_id, "payload:", payload)

    # check next intento number and enforce <=5
    try:
        next_num = get_next_intento_number(descripcion_id) if descripcion_id is not None else 1
    except Exception as e:
        print("Error leyendo intentos previos:", e)
        next_num = 1

    if next_num > 5:
        print(f"Ya se alcanzaron 5 intentos para descripcion_id={descripcion_id}. Ignorando.")
        # optionally update descripciones or publish an error result
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    start = time.time()
    try:
        res = predict_structured(payload.get("types", []), payload.get("color", ""), int(payload.get("height", 0)), top_k=3)
        elapsed_ms = int((time.time() - start) * 1000)
        # choose top candidate as pokemon_predicho
        top_candidate = res["results"][0] if res["results"] else {"name": None, "score": 0.0}
        pokemon_name = top_candidate.get("name")
        probability = float(top_candidate.get("score", 0.0))

        # insert intento row
        try:
            intento_id = insert_intento(descripcion_id, next_num, pokemon_name, probability, elapsed_ms)
            print(f"Intento registrado id={intento_id} (numero {next_num})")
        except Exception as e:
            print("Error insertando intento:", e)

        # publish result envelope
        result_obj = {"mode": res["mode"], "candidates": res["results"], "tiempo_ms": elapsed_ms, "intento_num": next_num}
        publish_result_to_queue(out_channel, descripcion_id, payload, result_obj)
    except Exception as e:
        print("Error durante predicción:", e)
        result_obj = {"mode": "error", "error": str(e)}
        publish_result_to_queue(out_channel, descripcion_id, payload, result_obj)

    ch.basic_ack(delivery_tag=method.delivery_tag)

print("Consumer matemático en espera...")
in_channel.basic_qos(prefetch_count=1)
in_channel.basic_consume(queue=QUEUE_IN, on_message_callback=on_message)
in_channel.start_consuming()
