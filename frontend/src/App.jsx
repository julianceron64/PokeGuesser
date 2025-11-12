import { useState } from "react";

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
  const [inputError, setInputError] = useState("");
  const [showTop3, setShowTop3] = useState(false); // 👈 NUEVO estado

  // === Enviar predicción inicial ===
  const handlePredict = async (e) => {
    e.preventDefault();
    setError(null);
    setResult(null);
    setDescripcionId(null);
    setDescription(null);
    setStatus("pending");
    setShowTop3(false);

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
    } catch {
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
    if (!pokemonName) {
      return "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/0.png";
    }
    const lower = pokemonName.toLowerCase().replace(/\s+/g, "-");
    return `https://img.pokemondb.net/sprites/home/normal/${lower}.png`;
  };

  // === Validaciones en tiempo real ===
  const handleTypesChange = (e) => {
    const value = e.target.value;
    if (/^[a-zA-Z, ]*$/.test(value)) {
      setTypes(value);
      setInputError("");
    } else {
      setInputError("Solo se permiten letras, comas y espacios en los tipos.");
    }
  };

  const handleColorChange = (e) => {
    const value = e.target.value;
    if (/^[a-zA-Z]*$/.test(value)) {
      setColor(value);
      setInputError("");
    } else {
      setInputError("El color solo puede contener letras.");
    }
  };

  // === Función para renderizar un Pokémon candidato ===
  const renderPokemonCandidate = (pokemon) => {
    const porcentaje = Math.min(pokemon.score * 100 / 3, 100).toFixed(2);
    const confiable = porcentaje >= 50;
    return (
      <div
        key={pokemon.name}
        style={{
          background: "#fff",
          borderRadius: "12px",
          boxShadow: "0 0 10px rgba(0,0,0,0.1)",
          padding: "1rem",
          marginBottom: "1rem",
          textAlign: "center",
        }}
      >
        <img
          src={getPokemonImage(pokemon.name)}
          alt={pokemon.name}
          style={{
            width: "100px",
            imageRendering: "pixelated",
            margin: "0.5rem auto",
          }}
          onError={(e) => {
            e.target.src =
              "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/0.png";
          }}
        />
        <h3 style={{ textTransform: "capitalize", margin: "0.5rem 0" }}>
          {pokemon.name}
        </h3>
        <p><strong>Precisión:</strong> {porcentaje}%</p>
        <p style={{ color: confiable ? "green" : "red", fontWeight: "600" }}>
          {confiable ? "✅ Predicción confiable" : "⚠️ Predicción no confiable"}
        </p>
      </div>
    );
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
            onChange={handleTypesChange}
            placeholder="Ejemplo: fire, flying"
            style={{
              ...inputStyle,
              borderColor: inputError.includes("tipos") ? "red" : "#ccc",
            }}
          />

          <label style={labelStyle}>Color:</label>
          <input
            type="text"
            value={color}
            onChange={handleColorChange}
            placeholder="Ejemplo: red"
            style={{
              ...inputStyle,
              borderColor: inputError.includes("color") ? "red" : "#ccc",
            }}
          />

          <label style={labelStyle}>Altura (en decímetros):</label>
          <input
            type="number"
            step="0.1"
            value={height}
            onChange={(e) => setHeight(e.target.value)}
            placeholder="Ejemplo: 17"
            style={inputStyle}
          />

          {inputError && (
            <p style={{ color: "red", fontSize: "0.85rem" }}>{inputError}</p>
          )}

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
            {status === "pending" && <p>⏳ Esperando resultado...</p>}
          </div>
        )}

        {/* === RESULTADOS === */}
        {status === "ready" &&
          result &&
          result.candidates &&
          result.candidates.length > 0 && (
            <div style={resultBox}>
              <h3>Prediccion realizada:</h3>
              {/* Top 1 siempre visible */}
              {renderPokemonCandidate(result.candidates[0])}

              {/* Botón para mostrar/ocultar top 3 */}
              {result.candidates.length > 1 && (
                <button
                  style={{
                    ...buttonStyle,
                    background: "#007bff",
                    marginTop: "0.5rem",
                  }}
                  onClick={() => setShowTop3(!showTop3)}
                  type="button"
                >
                  {showTop3 ? "Ocultar Top 3" : "Ver Top 3 Predicciones"}
                </button>
              )}

              {/* Mostrar top 3 si se activa */}
              {showTop3 &&
                result.candidates
                  .slice(0, 3)
                  .map((pokemon, index) => renderPokemonCandidate(pokemon))}
            </div>
          )}
      </div>
    </div>
  );
}

// === Estilos ===
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
  outline: "none",
  transition: "border-color 0.2s",
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
