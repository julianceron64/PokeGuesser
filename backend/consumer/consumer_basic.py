import pika
import json
import os
import time
from datetime import datetime

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")

LOG_FILE = "/app/consumer/received_messages.log"

def callback(ch, method, properties, body):
    mensaje = json.loads(body)
    descripcion = mensaje.get("descripcion", "")
    
    print(f"📨 Mensaje recibido: {descripcion}")
    
    # Simular procesamiento
    time.sleep(1)
    
    # Registrar en log (simulando persistencia)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} - Descripción recibida: {descripcion}\n")
    
    ch.basic_ack(delivery_tag=method.delivery_tag)

def main():
    print("📡 Conectando a RabbitMQ...")
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST)
    )
    channel = connection.channel()
    channel.queue_declare(queue="pokemon_descriptions", durable=True)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue="pokemon_descriptions", on_message_callback=callback
    )

    print(" Esperando mensajes... (Ctrl+C para salir)")
    channel.start_consuming()

if __name__ == "__main__":
    main()
