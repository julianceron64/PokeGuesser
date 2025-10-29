import pika
import json
import os

# Datos de conexión desde la variable de entorno o valor por defecto
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq")

# Conexión
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=RABBITMQ_HOST)
)
channel = connection.channel()

# Declarar la cola (si no existe)
channel.queue_declare(queue="pokemon_descriptions", durable=True)

def send_description(texto):
    mensaje = {"descripcion": texto}
    channel.basic_publish(
        exchange="",
        routing_key="pokemon_descriptions",
        body=json.dumps(mensaje),
        properties=pika.BasicProperties(
            delivery_mode=2  # hace el mensaje persistente
        ),
    )
    print(f" Mensaje enviado: {mensaje}")

# --- Prueba manual ---
if __name__ == "__main__":
    descripcion = input("Describe un Pokémon: ")
    send_description(descripcion)
    connection.close()
