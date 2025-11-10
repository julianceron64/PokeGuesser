#!/usr/bin/env python3
"""
Consumer (LangChain-style pipeline) que:
- usa retrieval (embeddings + cosine similarity) para recuperar top-10 pokemons
- pide a HuggingFace Inference API que razone sobre esos 10 y devuelva top-3 en JSON
- mantiene integraciones con RabbitMQ y SQLAlchemy igual que tu consumer original
- token HF en .env -> HUGGINGFACE_TOKEN
- modelo LLM HF en .env -> HUGGINGFACE_MODEL (por defecto: "tiiuae/falcon-7b-instruct" - puedes cambiar)
- (opcional) modelo de embeddings HF en .env -> HUGGINGFACE_EMBEDDING_MODEL
"""
import os
import json
import time
import math
import requests
import traceback
import numpy as np
import pika
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity

# Optional local fallback embedding
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ENV / Config
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/pokeguesser")
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")
QUEUE_IN = os.getenv("QUEUE_IN", "pokemon_descriptions")
QUEUE_OUT = os.getenv("QUEUE_OUT", "pokemon_predictions")
EMBEDDINGS_PATH = os.path.join("db", "pokemon_embeddings.npy")
EMBEDDING_MODEL_LOCAL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
HF_MODEL = os.getenv("HUGGINGFACE_MODEL", "tiiuae/falcon-7b-instruct")
HF_EMBEDDING_MODEL = os.getenv("HUGGINGFACE_EMBEDDING_MODEL")  # optional
TOP_RETRIEVE = int(os.getenv("TOP_RETRIEVE", "10"))
TOP_RETURN = int(os.getenv("TOP_RETURN", "3"))

engine = create_engine(DATABASE_URL)

# --- Load pokemons from DB once on startup ---
with engine.connect() as conn:
    pokemons = conn.execute(text(
        "SELECT id, name, type, color, habitat, height, weight, description FROM pokemons"
    )).fetchall()
print(f"[startup] Se cargaron {len(pokemons)} pokemons desde DB")

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
    try:
        return 1.0 / (1.0 + abs(int(user_h) - int(db_h)))
    except Exception:
        return 0.0

def build_text_for_embedding(p):
    # p is a SQLAlchemy row-like; use attributes by name
    parts = [getattr(p, "name", "") or "",
             getattr(p, "description", "") or "",
             getattr(p, "type", "") or "",
             getattr(p, "color", "") or "",
             getattr(p, "habitat", "") or "",
             f"height:{getattr(p, 'height', '')}" if getattr(p, 'height', None) is not None else ""]
    return " ".join([str(x) for x in parts if x])

pokemon_texts = [build_text_for_embedding(p) for p in pokemons]

# --- Embeddings management (HF Inference API optional, otherwise sentence-transformers fallback) ---
use_embeddings = False
embeddings = None

def hf_inference_embeddings(texts, model_name):
    """
    Call HF Inference API for embeddings. Accepts either a single string or list of strings.
    Returns numpy array (n, dim).
    """
    if HF_TOKEN is None:
        raise RuntimeError("HUGGINGFACE_TOKEN no está definido en el entorno para usar HF embeddings.")
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    # The HF embeddings models usually accept {"inputs": ["..."]} and return embeddings
    payload = {"inputs": texts}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"HF Embeddings request failed: {resp.status_code} {resp.text}")
    body = resp.json()
    # body expected to be list of embeddings or dict
    # normalize to 2D list
    if isinstance(body, dict) and "error" in body:
        raise RuntimeError(f"HF Embeddings error: {body}")
    # possible shapes: [{"embedding": [...]}, [...]] or [[...], [...]]
    emb_list = []
    if isinstance(body, list) and body and isinstance(body[0], dict) and "embedding" in body[0]:
        for item in body:
            emb_list.append(item["embedding"])
    elif isinstance(body, list) and body and isinstance(body[0], list):
        emb_list = body
    else:
        raise RuntimeError(f"Formato inesperado de embeddings HF: {body}")
    return np.array(emb_list, dtype=np.float32)

def ensure_embeddings():
    global use_embeddings, embeddings
    # try load from file first
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            embeddings = np.load(EMBEDDINGS_PATH)
            use_embeddings = True
            print("[embeddings] Cargadas desde archivo:", EMBEDDINGS_PATH)
            return
        except Exception as e:
            print("[embeddings] Error cargando .npy:", e)

    # try HF Inference embeddings if user provided model
    if HF_EMBEDDING_MODEL and HF_TOKEN:
        try:
            print("[embeddings] Generando embeddings via HuggingFace Inference API (model:", HF_EMBEDDING_MODEL,") ...")
            embeddings = hf_inference_embeddings(pokemon_texts, HF_EMBEDDING_MODEL)
            np.save(EMBEDDINGS_PATH, embeddings)
            use_embeddings = True
            print("[embeddings] Guardadas en:", EMBEDDINGS_PATH)
            return
        except Exception as e:
            print("[embeddings] HF inference embeddings falló:", e)
            # continue to fallback

    # fallback: sentence-transformers local
    if SentenceTransformer is not None:
        try:
            print("[embeddings] Generando embeddings con SentenceTransformer local:", EMBEDDING_MODEL_LOCAL)
            model = SentenceTransformer(EMBEDDING_MODEL_LOCAL)
            embeddings = model.encode(pokemon_texts, show_progress_bar=True, convert_to_numpy=True)
            np.save(EMBEDDINGS_PATH, embeddings)
            use_embeddings = True
            print("[embeddings] Guardadas en:", EMBEDDINGS_PATH)
            return
        except Exception as e:
            print("[embeddings] Fallback sentence-transformers falló:", e)

    # if still not available, mark false and rely on structured-only fallback
    use_embeddings = False
    embeddings = None
    print("[embeddings] No se pudieron obtener embeddings. El consumer usará reglas estructuradas cuando sea posible.")

ensure_embeddings()

# --- HF generation helper (calls HF Inference text-generation endpoint) ---
def hf_generate(prompt: str, model_name: str, timeout: int = 60):
    if HF_TOKEN is None:
        raise RuntimeError("HUGGINGFACE_TOKEN no está definido en el entorno para usar HF generation.")
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 512, "temperature": 0.0},
        "options": {"use_cache": False, "wait_for_model": True}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"HF generation failed: {resp.status_code} {resp.text}")
    body = resp.json()
    # try to extract generated_text
    if isinstance(body, dict) and "error" in body:
        raise RuntimeError(f"HF generation error: {body}")
    # Some models return [{"generated_text": "..."}]
    if isinstance(body, list) and len(body) > 0 and isinstance(body[0], dict) and "generated_text" in body[0]:
        return body[0]["generated_text"]
    # some models (inference) return {"generated_text": "..."}
    if isinstance(body, dict) and "generated_text" in body:
        return body["generated_text"]
    # otherwise, try to convert to str
    try:
        return json.dumps(body)
    except Exception:
        return str(body)

# --- RabbitMQ setup ---
credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
parameters = pika.ConnectionParameters(host=RABBITMQ_HOST, port=5672, credentials=credentials)
connection = pika.BlockingConnection(parameters)

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

# --- Retrieval + LLM chain ---
def retrieve_top_k_by_embedding(user_text: str, top_k: int = TOP_RETRIEVE):
    """
    Returns list of tuples (score, pokemon_index) sorted descending by similarity.
    """
    global embeddings
    if not use_embeddings or embeddings is None:
        return []

    # user embedding via HF Embeddings (if configured) else local sentence-transformers
    user_emb = None
    try:
        if HF_EMBEDDING_MODEL and HF_TOKEN:
            user_emb = hf_inference_embeddings([user_text], HF_EMBEDDING_MODEL)[0]
        else:
            # use local sentence-transformers if available
            if SentenceTransformer is None:
                raise RuntimeError("No hay SentenceTransformer disponible para embeddings locales.")
            model_local = SentenceTransformer(EMBEDDING_MODEL_LOCAL)
            user_emb = model_local.encode([user_text], convert_to_numpy=True)[0]
    except Exception as e:
        print("[retrieve] Error calculando embedding usuario:", e)
        return []

    sims = cosine_similarity([user_emb], embeddings)[0]
    idx_sorted = sims.argsort()[::-1][:top_k]
    return [(float(sims[i]), int(i)) for i in idx_sorted]

def build_prompt_for_llm(user_payload, retrieved_items):
    """
    Construye un prompt instructivo: se le dan al LLM los atributos de los N pokemons recuperados
    y se le pide que devuelva un JSON con top-3 candidatos, cada uno con id, name, score (0..1) y reason.
    """
    header = (
        "Eres un asistente que ayuda a identificar cuál Pokémon corresponde mejor a una descripción.\n"
        "Se te entregan datos de usuario (types, color, height) y una lista de candidatos recuperados de la base de datos.\n"
        "Analiza exclusivamente los candidatos proporcionados y devuelve **únicamente** un JSON con la clave \"results\" "
        "conteniendo un arreglo de hasta 3 objetos ordenados descendentemente por probabilidad.\n\n"
        "Formato de salida EXACTO (ejemplo):\n"
        '{ "mode": "langchain_hf", "results": ['
        '{"id": 1, "name":"Bulbasaur", "score": 0.95, "reason":"matching types and color"},'
        '{"id": 2, "name":"Ivysaur", "score": 0.5, "reason":"..."}'
        ']}\n\n'
        "Cada objeto debe tener: id (entero), name (string), score (float entre 0.0 y 1.0), reason (string breve).\n\n"
    )

    user_section = f"Usuario: types={json.dumps(user_payload.get('types', []))}, color={user_payload.get('color')}, height={user_payload.get('height')}\n\n"

    candidates_section = "Candidatos (base de datos):\n"
    for score, idx in retrieved_items:
        p = pokemons[idx]
        candidates_section += (
            f"- id: {p.id} | name: {p.name} | type: {p.type} | color: {p.color} | "
            f"habitat: {p.habitat} | height: {p.height} | weight: {p.weight}\n"
            f"  description: {((p.description or '')[:320]).replace(chr(10),' ')}\n"
        )

    instruction = (
        "\nInstrucción: Usando SOLO la información de los candidatos anteriores, "
        "devuelve el JSON especificado con los 3 mejores candidatos. No agregues texto extra, "
        "comentarios ni explicación. Si no estás seguro, asigna probabilidades bajas (p.ej. 0.05).\n"
        "Normaliza scores para que estén en [0,1]."
    )

    prompt = header + user_section + candidates_section + instruction
    return prompt

def predict_with_chain(payload):
    """
    Pipeline:
    - build user_text for retrieval
    - retrieve top-10 with embeddings
    - if retrieved empty => fallback to structured color filter method (like original)
    - build prompt with retrieved items
    - call HF Inference API LLM to get JSON results
    - parse and return results
    """
    user_types = payload.get("types", []) or []
    user_color = (payload.get("color") or "").strip().lower()
    user_height = int(payload.get("height", 0))

    # retrieval text
    user_text = " ".join(user_types + [user_color, f"{user_height}dm"])
    retrieved = retrieve_top_k_by_embedding(user_text, top_k=TOP_RETRIEVE)

    if not retrieved:
        # fallback: try structured color filter like the original
        candidates = [p for p in pokemons if (p.color or "").strip().lower() == user_color]
        if candidates:
            scored = []
            for p in candidates:
                db_types = parse_db_types(p.type)
                tscore = compute_type_score([t.lower() for t in user_types], db_types)
                hsim = height_similarity(user_height, p.height or 0)
                score = tscore + hsim
                scored.append((score, p))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for score, p in scored[:TOP_RETURN]:
                results.append({
                    "id": p.id,
                    "name": p.name,
                    "score": float(score),
                    "db_types": parse_db_types(p.type),
                    "db_color": p.color,
                    "db_height": p.height
                })
            return {"mode": "structured_color_filter", "results": results}

        return {"mode": "no_embeddings_no_color_candidates", "results": []}

    # build prompt
    prompt = build_prompt_for_llm({"types": user_types, "color": user_color, "height": user_height}, retrieved)

    # call HF LLM
    try:
        raw = hf_generate(prompt, HF_MODEL, timeout=120)
        # the model should return the JSON as text; try to parse JSON inside output
        # attempt 1: direct JSON parse
        raw_str = raw.strip()
        # sometimes the model returns text before/after JSON — try to find first '{' and last '}'
        first = raw_str.find('{')
        last = raw_str.rfind('}')
        if first != -1 and last != -1 and last > first:
            json_candidate = raw_str[first:last+1]
        else:
            json_candidate = raw_str

        parsed = json.loads(json_candidate)
        results = parsed.get("results") if isinstance(parsed, dict) else None
        if not results:
            # try another strategy: maybe the model returned a list directly
            if isinstance(parsed, list):
                results = parsed[:TOP_RETURN]
            else:
                raise ValueError("La respuesta del LLM no contiene 'results' o lista esperada.")
        # normalize & sanitize scores to [0,1]
        out_results = []
        for item in results[:TOP_RETURN]:
            try:
                sid = int(item.get("id"))
            except Exception:
                sid = None
            name = item.get("name")
            score = float(item.get("score", 0.0))
            score = max(0.0, min(1.0, score))
            reason = item.get("reason", "") if isinstance(item.get("reason", ""), str) else ""
            out_results.append({"id": sid, "name": name, "score": score, "reason": reason})
        return {"mode": "hf_inference_llm", "results": out_results}
    except Exception as e:
        print("[predict_with_chain] Error llamando al LLM HF:", e)
        traceback.print_exc()
        return {"mode": "error", "error": str(e), "raw_llm_output": raw if 'raw' in locals() else None}

# --- RabbitMQ message handler ---
def on_message(ch, method, properties, body):
    try:
        envelope = json.loads(body)
        descripcion_id = envelope.get("descripcion_id")
        payload = envelope.get("payload", {})
    except Exception as e:
        print("[on_message] Mensaje inválido:", e)
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    if descripcion_id is None:
        print("[on_message] Aviso: descripcion_id es None — el mensaje no fue creado por la API. Se procesará igualmente con descripcion_id NULL.")
    print("[on_message] Procesando descripcion_id:", descripcion_id, "payload:", payload)

    # check next intento number and enforce <=5
    try:
        next_num = get_next_intento_number(descripcion_id) if descripcion_id is not None else 1
    except Exception as e:
        print("[on_message] Error leyendo intentos previos:", e)
        next_num = 1

    if next_num > 5:
        print(f"[on_message] Ya se alcanzaron 5 intentos para descripcion_id={descripcion_id}. Ignorando.")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    start = time.time()
    try:
        res = predict_with_chain(payload)
        elapsed_ms = int((time.time() - start) * 1000)

        top_candidate = res.get("results", [{}])[0] if res.get("results") else {"name": None, "score": 0.0}
        pokemon_name = top_candidate.get("name")
        probability = float(top_candidate.get("score", 0.0) if top_candidate.get("score") is not None else 0.0)

        # insert intento row
        try:
            intento_id = insert_intento(descripcion_id, next_num, pokemon_name, probability, elapsed_ms)
            print(f"[on_message] Intento registrado id={intento_id} (numero {next_num})")
        except Exception as e:
            print("[on_message] Error insertando intento:", e)

        # publish result envelope
        result_obj = {"mode": res.get("mode"), "candidates": res.get("results", []), "tiempo_ms": elapsed_ms, "intento_num": next_num}
        publish_result_to_queue(out_channel, descripcion_id, payload, result_obj)
    except Exception as e:
        print("[on_message] Error durante predicción:", e)
        traceback.print_exc()
        result_obj = {"mode": "error", "error": str(e)}
        publish_result_to_queue(out_channel, descripcion_id, payload, result_obj)

    ch.basic_ack(delivery_tag=method.delivery_tag)

# --- Start consuming ---
print("[consumer] Consumer LangChain-style en espera...")
in_channel.basic_qos(prefetch_count=1)
in_channel.basic_consume(queue=QUEUE_IN, on_message_callback=on_message)
in_channel.start_consuming()

