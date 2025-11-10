import os
import json
import pika
import time

QUEUE_OUT = "pokemon_predictions"
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")

def start_result_listener(result_cache: dict):
    credentials = pika.PlainCredentials(
        os.getenv("RABBITMQ_USER", "user"),
        os.getenv("RABBITMQ_PASS", "password")
    )

    connection = None

    # --- REINTENTOS INFINITOS ---
    while connection is None:
        try:
            print("🔌 Intentando conectar a RabbitMQ...")
            params = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=5672,
                credentials=credentials
            )
            connection = pika.BlockingConnection(params)
            print("✅ Conectado a RabbitMQ!")
        except Exception as e:
            print("❌ No se pudo conectar a RabbitMQ, reintentando en 3s...", e)
            time.sleep(3)

    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_OUT, durable=True)
    print("🔁 Escuchando resultados en cola:", QUEUE_OUT)

    def callback(ch, method, properties, body):
        try:
            envelope = json.loads(body)
            desc_id = envelope.get("descripcion_id")
            if desc_id is not None:
                result_cache[desc_id] = envelope.get("result", {})
                print(f"✔️ Resultado actualizado para descripcion_id={desc_id}")
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print("⚠️ Error procesando mensaje de resultado:", e)
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_OUT, on_message_callback=callback)

    try:
        channel.start_consuming()
    except Exception as e:
        print("❌ Listener crasheó:", e)
