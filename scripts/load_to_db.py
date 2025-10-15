import os
import json
import sys

# 🔧 Asegura que se pueda importar 'backend' desde cualquier ubicación
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db.db import SessionLocal
from backend.db.models import Pokemon

def load_json_to_db():
    session = SessionLocal()

    # Ruta absoluta al archivo JSON
    data_path = os.path.join(os.path.dirname(__file__), "pokemon_data.json")

    if not os.path.exists(data_path):
        print(f" No se encontró el archivo: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        pokemons = json.load(f)

    print(f" Cargando {len(pokemons)} Pokémon en la base de datos...")

    for p in pokemons:
        try:
            # Convierte lista de tipos a string (ej: "fire, flying")
            types = ", ".join(p["types"]) if isinstance(p.get("types"), list) else p.get("types", "")

            pokemon = Pokemon(
                name=p.get("name", "Desconocido"),
                type=types,
                description=p.get("description", "Descripción no disponible.")
            )

            session.add(pokemon)

        except Exception as e:
            print(f" Error al procesar {p.get('name', 'desconocido')}: {e}")

    session.commit()
    session.close()
    print("✅ Todos los Pokémon fueron guardados correctamente en la base de datos.")

if __name__ == "__main__":
    load_json_to_db()
