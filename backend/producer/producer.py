#!/usr/bin/env python3
import pika
import json
import os
import uuid
import time

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")
QUEUE_NAME = "pokemon_descriptions"

def connect_channel():
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
    params = pika.ConnectionParameters(host=RABBITMQ_HOST, port=5672, credentials=credentials)
    while True:
        try:
            print("🔌 [Producer] Intentando conectar a RabbitMQ...")
            connection = pika.BlockingConnection(params)
            channel = connection.channel()
            channel.queue_declare(queue=QUEUE_NAME, durable=True)
            print(f"✅ [Producer] Conectado a RabbitMQ y cola '{QUEUE_NAME}' declarada!")
            return connection, channel
        except Exception as e:
            print(f"❌ [Producer] Error de conexión: {e}. Reintentando en 2s...")
            time.sleep(2)

def send_payload(channel, payload: dict):
    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=json.dumps(payload, ensure_ascii=False),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    print("✔️ [Producer] Mensaje enviado:", payload)

def manual_cli():
    connection, channel = connect_channel()
    print("\n💬 Escribe 'exit' para salir.")
    while True:
        types_raw = input("Types (coma-separado): ").strip()
        if types_raw.lower() == "exit":
            break
        types = [t.strip().lower() for t in types_raw.split(",") if t.strip()]
        color = input("Color: ").strip().lower()
        height_raw = input("Height (decímetros): ").strip()
        try:
            height = int(height_raw)
        except ValueError:
            print("❌ Height inválido. Debe ser entero.")
            continue

        payload = {"types": types, "color": color, "height": height}
        envelope = {"descripcion_id": str(uuid.uuid4()), "payload": payload}
        send_payload(channel, envelope)

    print("🔚 Cerrando conexión...")
    connection.close()

if __name__ == "__main__":
    manual_cli()
