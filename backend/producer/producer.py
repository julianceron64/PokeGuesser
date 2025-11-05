#!/usr/bin/env python3
import pika
import json
import os

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")
QUEUE_NAME = "pokemon_descriptions"

def connect_channel():
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=RABBITMQ_HOST))
    channel = connection.channel()
    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    return connection, channel

def send_payload(channel, payload: dict):
    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_NAME,
        body=json.dumps(payload, ensure_ascii=False),
        properties=pika.BasicProperties(delivery_mode=2)
    )
    print("✔️ Mensaje enviado:", payload)

def input_from_console():
    print("Introduce los datos para predecir (si 1 tipo, introduce solo uno).")
    types_raw = input("Types (coma-separado, ej: electric or fire,flying): ").strip()
    types = [t.strip().lower() for t in types_raw.split(",") if t.strip()] if types_raw else []
    color = input("Color (ej: yellow): ").strip().lower()
    height_raw = input("Height en decimetros (ej: 4 para 0.4m): ").strip()
    try:
        height = int(height_raw)
    except:
        print("Height inválido, se requiere entero en decímetros.")
        return None
    payload = {
        "types": types,
        "color": color,
        "height": height
    }
    return payload

if __name__ == "__main__":
    connection, channel = connect_channel()
    payload = input_from_console()
    if payload:
        send_payload(channel, payload)
    connection.close()
