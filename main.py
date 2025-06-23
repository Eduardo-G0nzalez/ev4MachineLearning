from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import joblib
import numpy as np

app = FastAPI()

# Cargar modelo y recursos
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# HTML embebido con selects amigables
html_form = """
<!DOCTYPE html>
<html>
<head>
  <title>Predicción de Supervivencia</title>
  <style>
    body {{
      margin: 0;
      padding: 30px 0;
      background-color: #0e0e0e;
      color: #f0f0f0;
      font-family: 'Segoe UI', sans-serif;
      display: flex;
      justify-content: center;
      min-height: 100vh;
      overflow-y: auto;
    }}
    .form-container {{
      background-color: #1a1a1a;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 0 20px rgba(0, 255, 153, 0.25);
      width: 800px;
      max-width: 95%;
    }}
    h2 {{
      text-align: center;
      color: #00ff99;
      margin-bottom: 20px;
    }}
    form {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    label {{
      font-weight: bold;
      margin-bottom: 6px;
      display: block;
    }}
    select {{
      width: 100%;
      padding: 8px;
      background-color: #2a2a2a;
      color: #fff;
      border: 1px solid #00ff99;
      border-radius: 4px;
    }}
    .full-width {{
      grid-column: span 2;
      text-align: center;
      margin-top: 20px;
    }}
    input[type=submit] {{
      background-color: #00ff99;
      color: #000;
      font-weight: bold;
      border: none;
      padding: 10px 20px;
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.3s;
    }}
    input[type=submit]:hover {{
      background-color: #00cc7a;
    }}
    .result {{
      margin-top: 25px;
      text-align: center;
      font-size: 18px;
    }}
  </style>
</head>
<body>
  <div class="form-container">
    <h2>🎮 Predicción de Supervivencia</h2>
    <form method="post">

      <label>Equipamiento Inicial:</label>
      <select name="RoundStartingEquipmentValue">
        <option value="500">Económica (solo pistola)</option>
        <option value="2000">Media compra (SMG/Scout)</option>
        <option value="3500">Compra casi completa</option>
        <option value="5000">Compra completa (rifle+granadas)</option>
      </select>

      <label>Asalto (rifle principal):</label>
      <select name="PrimaryAssaultRifle">
        <option value="1">Sí</option>
        <option value="0">No</option>
      </select>

      <label>Francotirador:</label>
      <select name="PrimarySniperRifle">
        <option value="1">Sí</option>
        <option value="0">No</option>
      </select>

      <label>Arma Pesada:</label>
      <select name="PrimaryHeavy">
        <option value="1">Sí</option>
        <option value="0">No</option>
      </select>

      <label>SMG:</label>
      <select name="PrimarySMG">
        <option value="1">Sí</option>
        <option value="0">No</option>
      </select>

      <label>Pistola:</label>
      <select name="PrimaryPistol">
        <option value="1">Sí</option>
        <option value="0">No</option>
      </select>

      <label>Kills en la Ronda:</label>
      <select name="RoundKills">
        <option value="0">0</option>
        <option value="1">1 kill</option>
        <option value="2">2 kills</option>
        <option value="3">3 o más</option>
      </select>

      <label>Asistencias:</label>
      <select name="RoundAssists">
        <option value="0">0</option>
        <option value="1">1 asistencia</option>
        <option value="2">2 o más</option>
      </select>

      <label>Headshots:</label>
      <select name="RoundHeadshots">
        <option value="0">0</option>
        <option value="1">1</option>
        <option value="2">2 o más</option>
      </select>

      <label>Flanqueos exitosos:</label>
      <select name="RoundFlankKills">
        <option value="0">0</option>
        <option value="1">1</option>
        <option value="2">2 o más</option>
      </select>

      <label>Mapa:</label>
      <select name="Map">
        <option value="0">Dust2</option>
        <option value="1">Inferno</option>
        <option value="2">Nuke</option>
        <option value="3">Mirage</option>
      </select>

      <label>Equipo:</label>
      <select name="Team">
        <option value="0">Terroristas</option>
        <option value="1">Antiterroristas</option>
      </select>

      <div class="full-width">
        <input type="submit" value="Predecir">
      </div>
    </form>
    <div class="result">
      {result}
    </div>
  </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return html_form.format(result="")

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request):
    form = await request.form()
    input_values = [float(form[f]) for f in features]
    data_scaled = scaler.transform([input_values])
    prediction = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0][1]

    resultado = f"<h3>🎯 ¿Sobrevive? {'<span style=\"color:lightgreen\">Sí</span>' if prediction == 1 else '<span style=\"color:orangered\">No</span>'}</h3>"
    resultado += f"<p>🔢 Probabilidad de supervivencia: {prob:.2%}</p>"

    return html_form.format(result=resultado)