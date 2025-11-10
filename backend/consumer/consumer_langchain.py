#!/usr/bin/env python3
"""
Consumer con LangChain (Versión B)
----------------------------------
Escucha mensajes de RabbitMQ (pokemon_descriptions),
recupera los Pokémon más similares por embeddings,
y usa un modelo de lenguaje de Hugging Face para predecir
los 3 más probables en formato JSON.
"""

import os
import json
import pika
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------- CONFIG ----------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/pokeguesser")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")
QUEUE_IN = os.getenv("QUEUE_IN", "pokemon_descriptions")
QUEUE_OUT = os.getenv("QUEUE_OUT", "pokemon_predictions")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL", "google/gemma-2b-it")

# ---------------- DB CONNECTION ----------------
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    pokemons = conn.execute(text("SELECT id, name, type, color, description FROM pokemons")).fetchall()
print(f"[startup] Se cargaron {len(pokemons)} Pokémon desde la base de datos.")

# ---------------- EMBEDDINGS ----------------
print("[embeddings] Generando embeddings locales (sentence-transformers)...")
texts = [f"{p.name} {p.description} {p.type} {p.color}" for p in pokemons]
model_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model_emb.encode(texts, convert_to_numpy=True)

# ---------------- LANGCHAIN SETUP ----------------
print("[langchain] Inicializando modelo:", HF_MODEL)
llm = HuggingFaceEndpoint(
    repo_id=HF_MODEL,
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation"
)

template = """
Eres un asistente que identifica Pokémon basándote en descripciones.
Datos del usuario:
- Tipos: {types}
- Color: {color}
- Altura: {height}

Lista de candidatos:
{candidates}

Devuelve un JSON EXACTO con el formato:
{{"results":[{{"id":int,"name":str,"score":float,"reason":str}}]}}
"""

prompt = PromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt)

# ---------------- RABBITMQ ----------------
credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials))
in_channel = connection.channel()
out_channel = connection.channel()
in_channel.queue_declare(queue=QUEUE_IN, durable=True)
out_channel.queue_declare(queue=QUEUE_OUT, durable=True)

# ---------------- FUNCIONES AUXILIARES ----------------
def retrieve_candidates(payload):
    text = " ".join(payload.get("types", [])) + " " + (payload.get("color") or "")
    emb = model_emb.encode([text], convert_to_numpy=True)[0]
    sims = cosine_similarity([emb], embeddings)[0]
    top = np.argsort(sims)[::-1][:10]
    return [pokemons[i] for i in top]

def on_message(ch, method, properties, body):
    msg = json.loads(body)
    descripcion_id = msg.get("descripcion_id")
    payload = msg.get("payload", {})
    print(f"[worker] Procesando descripción {descripcion_id} -> {payload}")

    try:
        candidates = retrieve_candidates(payload)
        candidates_txt = "\n".join([f"id:{p.id}, name:{p.name}, type:{p.type}, color:{p.color}" for p in candidates])
        result = chain.invoke({
            "types": payload.get("types"),
            "color": payload.get("color"),
            "height": payload.get("height"),
            "candidates": candidates_txt
        })

        out_msg = {
            "descripcion_id": descripcion_id,
            "payload": payload,
            "result": result
        }
        out_channel.basic_publish(
            exchange="", routing_key=QUEUE_OUT,
            body=json.dumps(out_msg),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        print(f"[worker] Resultado publicado para descripción {descripcion_id}")
    except Exception as e:
        print("[worker] Error procesando mensaje:", e)
    ch.basic_ack(delivery_tag=method.delivery_tag)

print("[worker] Esperando mensajes...")
in_channel.basic_qos(prefetch_count=1)
in_channel.basic_consume(queue=QUEUE_IN, on_message_callback=on_message)
in_channel.start_consuming()
#!/usr/bin/env python3
"""
Consumer con LangChain (Versión B)
----------------------------------
Escucha mensajes de RabbitMQ (pokemon_descriptions),
recupera los Pokémon más similares por embeddings,
y usa un modelo de lenguaje de Hugging Face para predecir
los 3 más probables en formato JSON.
"""

import os
import json
import pika
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ---------------- CONFIG ----------------
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/pokeguesser")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")
QUEUE_IN = os.getenv("QUEUE_IN", "pokemon_descriptions")
QUEUE_OUT = os.getenv("QUEUE_OUT", "pokemon_predictions")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL", "google/gemma-2b-it")

# ---------------- DB CONNECTION ----------------
engine = create_engine(DATABASE_URL)

with engine.connect() as conn:
    pokemons = conn.execute(text("SELECT id, name, type, color, description FROM pokemons")).fetchall()
print(f"[startup] Se cargaron {len(pokemons)} Pokémon desde la base de datos.")

# ---------------- EMBEDDINGS ----------------
print("[embeddings] Generando embeddings locales (sentence-transformers)...")
texts = [f"{p.name} {p.description} {p.type} {p.color}" for p in pokemons]
model_emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model_emb.encode(texts, convert_to_numpy=True)

# ---------------- LANGCHAIN SETUP ----------------
print("[langchain] Inicializando modelo:", HF_MODEL)
llm = HuggingFaceEndpoint(
    repo_id=HF_MODEL,
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation"
)

template = """
Eres un asistente que identifica Pokémon basándote en descripciones.
Datos del usuario:
- Tipos: {types}
- Color: {color}
- Altura: {height}

Lista de candidatos:
{candidates}

Devuelve un JSON EXACTO con el formato:
{{"results":[{{"id":int,"name":str,"score":float,"reason":str}}]}}
"""

prompt = PromptTemplate.from_template(template)
chain = LLMChain(llm=llm, prompt=prompt)

# ---------------- RABBITMQ ----------------
credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST, credentials=credentials))
in_channel = connection.channel()
out_channel = connection.channel()
in_channel.queue_declare(queue=QUEUE_IN, durable=True)
out_channel.queue_declare(queue=QUEUE_OUT, durable=True)

# ---------------- FUNCIONES AUXILIARES ----------------
def retrieve_candidates(payload):
    text = " ".join(payload.get("types", [])) + " " + (payload.get("color") or "")
    emb = model_emb.encode([text], convert_to_numpy=True)[0]
    sims = cosine_similarity([emb], embeddings)[0]
    top = np.argsort(sims)[::-1][:10]
    return [pokemons[i] for i in top]

def on_message(ch, method, properties, body):
    msg = json.loads(body)
    descripcion_id = msg.get("descripcion_id")
    payload = msg.get("payload", {})
    print(f"[worker] Procesando descripción {descripcion_id} -> {payload}")

    try:
        candidates = retrieve_candidates(payload)
        candidates_txt = "\n".join([f"id:{p.id}, name:{p.name}, type:{p.type}, color:{p.color}" for p in candidates])
        result = chain.invoke({
            "types": payload.get("types"),
            "color": payload.get("color"),
            "height": payload.get("height"),
            "candidates": candidates_txt
        })

        out_msg = {
            "descripcion_id": descripcion_id,
            "payload": payload,
            "result": result
        }
        out_channel.basic_publish(
            exchange="", routing_key=QUEUE_OUT,
            body=json.dumps(out_msg),
            properties=pika.BasicProperties(delivery_mode=2)
        )
        print(f"[worker] Resultado publicado para descripción {descripcion_id}")
    except Exception as e:
        print("[worker] Error procesando mensaje:", e)
    ch.basic_ack(delivery_tag=method.delivery_tag)

print("[worker] Esperando mensajes...")
in_channel.basic_qos(prefetch_count=1)
in_channel.basic_consume(queue=QUEUE_IN, on_message_callback=on_message)
in_channel.start_consuming()
