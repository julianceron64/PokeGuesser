import os
import json
import pika
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/pokeguesser")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


EMBEDDINGS_PATH = os.path.join("db", "pokemon_embeddings.npy")


print(" Conectando a la base de datos...")
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    pokemons = conn.execute(text("SELECT id, name, type, color, habitat, description FROM pokemons")).fetchall()

print(f" Se cargaron {len(pokemons)} Pokémon desde la base de datos.")

model = SentenceTransformer(MODEL_NAME)

def build_description(p):
    """Crea un texto descriptivo combinando varios campos."""
    parts = [p.description or "", p.type or "", p.color or "", p.habitat or ""]
    return " ".join(filter(None, parts))

pokemon_texts = [build_description(p) for p in pokemons]


os.makedirs("db", exist_ok=True)

if os.path.exists(EMBEDDINGS_PATH):
    print(" Cargando embeddings existentes desde archivo...")
    embeddings = np.load(EMBEDDINGS_PATH)
else:
    print(" Generando embeddings por primera vez...")
    embeddings = model.encode(pokemon_texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f" Embeddings guardados en '{EMBEDDINGS_PATH}'.")


def predict_pokemon(descripcion, top_k=3):
    desc_emb = model.encode([descripcion], convert_to_numpy=True)
    sims = cosine_similarity(desc_emb, embeddings)[0]
    idx_sorted = sims.argsort()[::-1][:top_k]

    results = []
    for i in idx_sorted:
        p = pokemons[i]
        results.append({
            "id": p.id,
            "name": p.name,
            "score": float(sims[i]),
            "description": p.description,
        })
    return results


def callback(ch, method, properties, body):
    data = json.loads(body)
    descripcion = data.get("descripcion", "")
    print(f"\n Descripción recibida: {descripcion}")

    try:
        results = predict_pokemon(descripcion)
        print(" Resultados:")
        for r in results:
            print(f" - {r['name']} (similitud={r['score']:.3f})")
    except Exception as e:
        print(" Error al procesar descripción:", e)

    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    print(" Conectando a RabbitMQ...")
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue="pokemon_descriptions", durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue="pokemon_descriptions", on_message_callback=callback)
    print(" IA en espera de descripciones...")
    channel.start_consuming()

if __name__ == "__main__":
    main()
