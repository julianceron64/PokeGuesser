#!/usr/bin/env python3
import pika
import json
import os
import uuid
import time  # ← necesario para los reintentos

# Variables de entorno y configuración
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "user")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "password")
QUEUE_NAME = "pokemon_descriptions"  # ← nombre de la cola unificado con el consumer

def connect_channel():
    """Conecta a RabbitMQ con reintentos infinitos."""
    credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)

    params = pika.ConnectionParameters(
        host=RABBITMQ_HOST,
        port=5672,
        credentials=credentials
    )

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
    """Envía un mensaje JSON al broker."""
    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=json.dumps(payload, ensure_ascii=False),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    print("✔️ [Producer] Mensaje enviado:", payload)

def manual_cli():
    """CLI manual para probar envío de mensajes."""
    print("Introduce los datos (si 1 tipo, introduce solo uno).")
    types_raw = input("Types (coma-separado, ej: electric or fire,flying): ").strip()
    types = [t.strip().lower() for t in types_raw.split(",") if t.strip()] if types_raw else []
    color = input("Color (ej: yellow): ").strip().lower()
    height_raw = input("Height en decimetros (ej: 4 para 0.4m): ").strip()

    try:
        height = int(height_raw)
    except ValueError:
        print("Height inválido, se requiere entero en decímetros.")
        return None

    payload = {"types": types, "color": color, "height": height}
    envelope = {"descripcion_id": str(uuid.uuid4()), "payload": payload}

    connection, channel = connect_channel()
    send_payload(channel, envelope)
    connection.close()

if __name__ == "__main__":
    manual_cli()
