import requests
import json
from time import sleep

POKEAPI_BASE_URL = "https://pokeapi.co/api/v2"
OUTPUT_FILE = "pokemon_data.json"

def get_pokemon_data(pokemon_id):
    """Obtiene los datos básicos y la descripción del Pokémon."""
    try:
        # Datos principales
        pokemon_url = f"{POKEAPI_BASE_URL}/pokemon/{pokemon_id}"
        response = requests.get(pokemon_url)
        response.raise_for_status()
        pokemon = response.json()

        # Datos de especie (para descripción y color)
        species_url = f"{POKEAPI_BASE_URL}/pokemon-species/{pokemon_id}"
        response = requests.get(species_url)
        response.raise_for_status()
        species = response.json()

        # Tomar el flavor text en español o inglés (prioriza español)
        flavor_entries = species.get("flavor_text_entries", [])
        description = next(
            (entry["flavor_text"].replace("\n", " ").replace("\f", " ")
             for entry in flavor_entries if entry["language"]["name"] == "es"),
            next(
                (entry["flavor_text"].replace("\n", " ").replace("\f", " ")
                 for entry in flavor_entries if entry["language"]["name"] == "en"),
                "Descripción no disponible."
            )
        )

        # Estructura de salida
        data = {
            "id": pokemon["id"],
            "name": pokemon["name"],
            "height": pokemon["height"],
            "weight": pokemon["weight"],
            "types": [t["type"]["name"] for t in pokemon["types"]],
            "color": species.get("color", {}).get("name") if species.get("color") else "unknown",
            "habitat": species.get("habitat", {}).get("name") if species.get("habitat") else "unknown",
            "description": description
        }

        return data

    except Exception as e:
        print(f"Error con Pokémon ID {pokemon_id}: {e}")
        return None
        
    except Exception as e:
    	print(f"⚠️ Error con Pokémon ID {pokemon_id}: {e}")
    	sleep(1)
    	return None



def main():
    all_pokemon = []
    total_pokemon = 1025  # Todas las generaciones hasta Paldea

    print(f"Descargando información de {total_pokemon} Pokémon...")

    for i in range(1, total_pokemon + 1):
        data = get_pokemon_data(i)
        if data:
            all_pokemon.append(data)
        sleep(0.3)  # Pausa ligera para no saturar la API
        print(f"✔️ Pokémon {i} descargado: {data['name'] if data else 'error'}")

    # Guardar archivo JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_pokemon, f, ensure_ascii=False, indent=4)

    print(f"\n Datos guardados en {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
