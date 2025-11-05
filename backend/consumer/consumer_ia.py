#!/usr/bin/env python3
import os
import json
import pika
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity

# Embeddings model only used for fallback:
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
model = SentenceTransformer(MODEL_NAME)



DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/pokeguesser")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
QUEUE_NAME = "pokemon_descriptions"
EMBEDDINGS_PATH = os.path.join("db", "pokemon_embeddings.npy")


# Height tolerance (we use continuous score formula; tolerance used conceptually)
HEIGHT_TOLERANCE_DM = 2  # ±2 decimeters (informational, scoring is continuous)

# --------------------
print(" Conectando a la base de datos...")
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    pokemons = conn.execute(text("SELECT id, name, type, color, habitat, height, weight, description FROM pokemons")).fetchall()

print(f" Se cargaron {len(pokemons)} Pokémon desde la base de datos.")

# Precompute textual descriptions used for fallback embeddings
def build_text_for_embedding(p):
    parts = [p.name or "", p.description or "", p.type or "", p.color or "", p.habitat or ""]
    return " ".join([str(x) for x in parts if x])

pokemon_texts = [build_text_for_embedding(p) for p in pokemons]

# Try to load precomputed embeddings for fallback; generate if missing
use_embeddings = False
if os.path.exists(EMBEDDINGS_PATH):
    try:
        embeddings = np.load(EMBEDDINGS_PATH)
        use_embeddings = True
        print(" Embeddings cargados desde archivo (fallback listo).")
    except Exception as e:
        print(" Error cargando embeddings:", e)
        use_embeddings = False

# If no embeddings saved, lazily prepare model but do NOT encode yet
if not use_embeddings:
    model = SentenceTransformer(EMBEDDING_MODEL)
    # we will encode only if fallback occurs to save time/ram

# --------------------
def parse_db_types(type_field: str):
    if not type_field:
        return []
    # db stored like "fire, flying" or "fire"
    parts = [t.strip().lower() for t in type_field.split(",") if t.strip()]
    return parts

def compute_type_score(user_types, db_types):
    # user_types and db_types are lists of lowercase strings
    if not user_types:
        return 0
    # Count matches
    matches = sum(1 for t in user_types if t in db_types)
    # scoring rules: 2 if both match, 1 if one matches, 0 otherwise
    if len(user_types) >= 2:
        return 2 if matches == len(user_types) else (1 if matches >= 1 else 0)
    else:
        return 1 if matches >= 1 else 0

def height_similarity(user_h, db_h):
    # continuous similarity: 1 / (1 + abs(diff))
    return 1.0 / (1.0 + abs(int(user_h) - int(db_h)))

def predict_structured(user_types, user_color, user_height, top_k=3):
    user_color = (user_color or "").strip().lower()
    user_types = [t.lower() for t in user_types] if user_types else []

    # Stage 1: candidates with exact color match
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

    # Stage 2: no color match — fallback to embeddings similarity using textual proxy
    print("⚠️ No hay candidatos con color exacto. Usando fallback por embeddings (semantic).")
    # ensure embeddings exist
    global use_embeddings, embeddings, model
    if not use_embeddings:
        print(" Generando embeddings para fallback (esto puede tardar la primera vez)...")
        embeddings = model.encode(pokemon_texts, show_progress_bar=True, convert_to_numpy=True)
        np.save(EMBEDDINGS_PATH, embeddings)
        use_embeddings = True
        print(" Embeddings generados y guardados para futuros fallbacks.")
    # build user pseudo-text from structured fields to use embeddings
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

# --------------------
def callback(ch, method, properties, body):
    try:
        data = json.loads(body)
    except Exception as e:
        print(" Mensaje inválido (no JSON):", e)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    user_types = data.get("types", [])
    user_color = data.get("color", "")
    user_height = data.get("height", None)

    print("\n Descripción estructurada recibida:", {"types": user_types, "color": user_color, "height": user_height})

    # validation
    if user_height is None or user_color is None or user_types is None:
        print(" Payload incompleto. Se requiere 'types','color','height'.")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    try:
        res = predict_structured(user_types, user_color, int(user_height), top_k=3)
        print("🔎 Modo:", res["mode"])
        for r in res["results"]:
            print(f" - {r['name']} (score={r['score']:.4f}) types={r.get('db_types')} color={r.get('db_color')} height={r.get('db_height')}")
    except Exception as e:
        print(" Error en predicción:", e)

    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    print(" Conectando a RabbitMQ...")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=callback)
    print(" Consumer en espera de mensajes...")
    channel.start_consuming()

if __name__ == "__main__":
    main()
