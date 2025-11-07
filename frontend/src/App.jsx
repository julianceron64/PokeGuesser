import { useState, useEffect } from "react";

function App() {
  const [types, setTypes] = useState("");
  const [color, setColor] = useState("");
  const [height, setHeight] = useState("");
  const [descripcionId, setDescripcionId] = useState(null);
  const [description, setDescription] = useState(null);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // === Enviar predicción inicial ===
  const handlePredict = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setDescripcionId(null);
    setDescription(null);
    setStatus("pending");

    // Validaciones simples
    if (!types.trim() || !color.trim() || !height.trim()) {
      setError("Por favor completa todos los campos.");
      return;
    }

    const heightNum = parseFloat(height);
    if (isNaN(heightNum) || heightNum <= 0) {
      setError("La altura debe ser un número positivo.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          types: types.split(",").map((t) => t.trim()),
          color,
          height: heightNum,
        }),
      });

      const data = await response.json();

      if (data.descripcion_id) {
        setDescripcionId(data.descripcion_id);
        fetchDescription(data.descripcion_id);
        pollResult(data.descripcion_id);
      } else {
        setError("No se recibió un ID de descripción del backend.");
      }
    } catch (err) {
      setError("Error al conectar con el backend.");
    } finally {
      setLoading(false);
    }
  };

  // === Obtener descripción original ===
  const fetchDescription = async (id) => {
    try {
      const res = await fetch(`/api/description/${id}`);
      const data = await res.json();
      setDescription(data);
    } catch (err) {
      console.error("Error obteniendo descripción:", err);
    }
  };

  // === Hacer polling al resultado ===
  const pollResult = async (id) => {
    const interval = setInterval(async () => {
      const res = await fetch(`/api/predict/result/${id}`);
      const data = await res.json();
      if (data.status === "ready") {
        setStatus("ready");
        setResult(data.result);
        clearInterval(interval);
      }
    }, 2000);
  };

  // === Obtener imagen del Pokémon predicho ===
  const getPokemonImage = (pokemonName) => {
    if (!pokemonName) return "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/0.png";
    const lower = pokemonName.toLowerCase();
    // fallback genérico
    return `https://img.pokemondb.net/sprites/home/normal/${lower}.png`;
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "linear-gradient(160deg, #e8f0f8, #f8fafc)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        fontFamily: "Poppins, sans-serif",
        padding: "2rem",
      }}
    >
      <div
        style={{
          background: "white",
          borderRadius: "20px",
          boxShadow: "0 8px 25px rgba(0,0,0,0.1)",
          width: "400px",
          padding: "2rem",
          textAlign: "center",
        }}
      >
        <h1 style={{ color: "#333", marginBottom: "1rem" }}>PokeGuesser</h1>

        <form onSubmit={handlePredict}>
          <label style={labelStyle}>Tipo(s):</label>
          <input
            type="text"
            value={types}
            onChange={(e) => setTypes(e.target.value)}
            placeholder="Ejemplo: fire, flying"
            style={inputStyle}
          />

          <label style={labelStyle}>Color:</label>
          <input
            type="text"
            value={color}
            onChange={(e) => setColor(e.target.value)}
            placeholder="Ejemplo: red"
            style={inputStyle}
          />

          <label style={labelStyle}>Altura:</label>
          <input
            type="number"
            value={height}
            onChange={(e) => setHeight(e.target.value)}
            placeholder="Ejemplo: 12"
            style={inputStyle}
          />

          <button
            type="submit"
            disabled={loading}
            style={{
              ...buttonStyle,
              background: loading ? "#ccc" : "#ff4747",
            }}
          >
            {loading ? "Enviando..." : "Enviar Predicción"}
          </button>
        </form>

        {error && <p style={{ color: "red", marginTop: "1rem" }}>{error}</p>}

        {descripcionId && (
          <div style={resultBox}>
            <p><strong>ID Descripción:</strong> {descripcionId}</p>
            {description && (
              <div style={{ fontSize: "0.9rem", color: "#555" }}>
                <p><strong>Texto:</strong> {JSON.stringify(description.texto)}</p>
                <p><strong>Fecha:</strong> {description.fecha_envio}</p>
              </div>
            )}
            {status === "pending" && <p>⏳ Esperando resultado...</p>}
          </div>
        )}

        {status === "ready" && result && (
          <div style={resultBox}>
            <h3>Resultado del Backend</h3>
            <img
              src={getPokemonImage(result.pokemon_predicho)}
              alt="Pokémon predicho"
              style={{
                width: "120px",
                imageRendering: "pixelated",
                margin: "0.5rem auto",
              }}
              onError={(e) => {
                e.target.src = "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/0.png";
              }}
            />
            <pre
              style={{
                background: "#f6f8fa",
                padding: "1rem",
                borderRadius: "8px",
                textAlign: "left",
                fontSize: "0.85rem",
                overflowX: "auto",
              }}
            >
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

const labelStyle = {
  display: "block",
  textAlign: "left",
  marginBottom: "0.3rem",
  color: "#444",
  fontWeight: "600",
  fontSize: "0.9rem",
};

const inputStyle = {
  width: "100%",
  padding: "0.6rem",
  borderRadius: "8px",
  border: "1px solid #ccc",
  marginBottom: "1rem",
  fontSize: "0.9rem",
};

const buttonStyle = {
  width: "100%",
  padding: "0.8rem",
  color: "white",
  fontWeight: "600",
  border: "none",
  borderRadius: "10px",
  cursor: "pointer",
};

const resultBox = {
  marginTop: "1.5rem",
  background: "#fafafa",
  borderRadius: "12px",
  padding: "1rem",
  boxShadow: "0 0 5px rgba(0,0,0,0.05)",
  textAlign: "left",
};

export default App;
