CREATE TABLE IF NOT EXISTS pokemons (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    type VARCHAR(50),
    color VARCHAR(30),
    habitat VARCHAR(50),
    height INTEGER,
    weight INTEGER,
    description TEXT
);
