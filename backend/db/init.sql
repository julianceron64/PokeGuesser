


CREATE TABLE IF NOT EXISTS pokemons (
     id SERIAL PRIMARY KEY,
     name VARCHAR(100) NOT NULL,
     type VARCHAR(100),
     color VARCHAR(50),
     habitat VARCHAR(100),
     height INTEGER,
     weight INTEGER,
     description TEXT
 );


CREATE TABLE IF NOT EXISTS descripciones (
    id SERIAL PRIMARY KEY,
    texto TEXT NOT NULL,
    fecha_envio TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE TABLE IF NOT EXISTS intentos (
    id SERIAL PRIMARY KEY,
    descripcion_id INTEGER REFERENCES descripciones(id) ON DELETE CASCADE,
    numero_intento INTEGER NOT NULL CHECK (numero_intento >= 1 AND numero_intento <= 5),
    pokemon_predicho VARCHAR(100),
    es_correcto BOOLEAN DEFAULT FALSE,
    probabilidad FLOAT,
    tiempo_respuesta_ms INTEGER,
    fecha_intento TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


CREATE INDEX IF NOT EXISTS idx_intentos_descripcion ON intentos(descripcion_id);
CREATE INDEX IF NOT EXISTS idx_pokemon_name ON pokemons(name);
