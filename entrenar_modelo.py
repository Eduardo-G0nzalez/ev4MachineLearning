import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# --- Cargar y preparar los datos ---
df = pd.read_csv("Anexo.csv", sep=";", low_memory=False)

# Eliminar columnas irrelevantes
df = df.drop(columns=[
    "Unnamed: 0", "MatchId", "RoundId", "FirstKillTime", "TimeAlive",
    "TravelledDistance", "RoundWinner", "MatchWinner", "AbnormalMatch"
], errors="ignore")

# Asegurar que el objetivo esté bien definido
df["Survived"] = df["Survived"].astype(int)

# Codificar variables categóricas
df["Map"] = LabelEncoder().fit_transform(df["Map"].astype(str))
df["Team"] = LabelEncoder().fit_transform(df["Team"].astype(str))

# --- Definir features a usar (coinciden con el HTML) ---
features = [
    "RoundStartingEquipmentValue",
    "PrimaryAssaultRifle",
    "PrimarySniperRifle",
    "PrimaryHeavy",
    "PrimarySMG",
    "PrimaryPistol",
    "RoundKills",
    "RoundAssists",
    "RoundHeadshots",
    "RoundFlankKills",
    "Map",
    "Team"
]

X = df[features]
y = df["Survived"]

# Escalado
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División de datos
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

# Entrenamiento del modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo entrenado y recursos
joblib.dump(model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(features, "features.pkl")

print("✅ Modelo entrenado correctamente con 12 features.")
print("💾 Archivos guardados: random_forest_model.pkl, scaler.pkl, features.pkl")