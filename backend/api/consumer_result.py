import os
import json
import pika

QUEUE_OUT = "pokemon_predictions"
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")

def start_result_listener(result_cache: dict):
    credentials = pika.PlainCredentials(
        os.getenv("RABBITMQ_USER", "user"),
        os.getenv("RABBITMQ_PASS", "password")
    )
    params = pika.ConnectionParameters(host=RABBITMQ_HOST, port=5672, credentials=credentials)
    connection = pika.BlockingConnection(params)
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
            print("Error procesando mensaje de resultado:", e)
            ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_OUT, on_message_callback=callback)
    channel.start_consuming()
