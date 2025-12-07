from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

app = Flask(__name__)


df=pd.read_csv('Students Social Media Addiction_enriched.csv')

df = df.drop_duplicates().copy()

df = df.dropna(how="all")




X_ADDITC = df[['Age','Gender','Avg_Daily_Usage_Hours','Sleep_Hours_Per_Night']].copy()



###################### PREDICCION SCORE DE ADDICCION ####################

y_adiccion = df['Addicted_Score']

# Codificacion de la columna genero
X_ENCODED_ADDICT= pd.get_dummies(X_ADDITC, columns=["Gender"], drop_first=True)

X_train_addict, X_test_addict, y_train_addict, y_test_addict = train_test_split(
    X_ENCODED_ADDICT, 
    y_adiccion, 
    test_size=0.2, 
    random_state=42
)

rf_addiction = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42)
rf_addiction.fit(X_train_addict, y_train_addict)

# 4. Evaluar en el conjunto de prueba
train_score_addiction = rf_addiction.score(X_train_addict, y_train_addict)
test_score_addiction = rf_addiction.score(X_test_addict, y_test_addict)

print(f"Precisión en entrenamiento score de addicion: {train_score_addiction:.2f}")
print(f"Precisión en prueba score de addicion:        {test_score_addiction:.2f}")


###################### PREDICCION SCORE DE SALUD  ####################


y_salud_mental = df['Mental_Health_Score']

# Codificacion de la columna genero
X_ENCODED_MENTAL = X_ENCODED_ADDICT.copy()
X_ENCODED_MENTAL["Addicted_Score"] = df['Addicted_Score']


X_train_mental, X_test_mental, y_train_mental, y_test_mental = train_test_split(
    X_ENCODED_MENTAL, 
    y_salud_mental, 
    test_size=0.2, 
    random_state=42
)


rf_mental = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42)
rf_mental.fit(X_train_mental, y_train_mental)

# 4. Evaluar en el conjunto de prueba
train_score_mental =  rf_mental.score(X_train_mental, y_train_mental)
test_score_mental =  rf_mental.score(X_test_mental, y_test_mental)

print(f"Precisión en entrenamiento score de salud mental: {train_score_mental:.2f}")
print(f"Precisión en prueba score de salud mental:        {test_score_mental:.2f}")



new_student = [[19, 4.5, 5.0, True]]
predict_score_addiction = rf_addiction.predict(new_student)[0]
print("Prediccion Score de adiccion: ", predict_score_addiction)



new_student = [new_student[0] + [float(predict_score_addiction)]]
predict_score_mental = rf_mental.predict(new_student)[0]
print("Prediccion Score de Salud mental: ", predict_score_mental)


#########################  PARTE DE YLEM  ###############################

# ======================  MODELO 3  ======================
# Clasificador de Nivel de Riesgo de Adicción (BAJO/MEDIO/ALTO)

df["Addiction_Risk_Level"] = pd.cut(
    df["Addicted_Score"],
    bins=[0, 3.4, 6.6, 10],
    labels=["BAJO", "MEDIO", "ALTO"]
)

y_risk_level = df["addiction_risk_level"]
X_LEVEL = X_ENCODED_MENTAL.copy()

# Entrenamiento
X_train_level, X_test_level, y_train_level, y_test_level = train_test_split(
    X_LEVEL, 
    y_risk_level, 
    test_size=0.2, 
    random_state=42
)

rf_addiction_level = RandomForestClassifier(n_estimators=100, random_state=42)
rf_addiction_level.fit(X_train_level, y_train_level)

print("\nModelo 3 (Nivel Adicción) entrenado correctamente!")


# ======================  MODELO 4  ======================
# Clasificador de Riesgo Académico (ALTO/BAJO)

y_risk = df["academic_risk_flag"]
X_RISK = X_ENCODED_MENTAL.copy()

X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
    X_RISK, 
    y_risk, 
    test_size=0.2, 
    random_state=42
)

rf_risk = RandomForestClassifier(n_estimators=120, max_features="sqrt", random_state=42)
rf_risk.fit(X_train_risk, y_train_risk)



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form["age"])
    usage = float(request.form["usage"])
    sleep = float(request.form["sleep"])
    gender = request.form["gender"].strip().lower()

    gender_encoded = 1 if gender == "male" else 0

    # ----------- Modelo 1 ---------
    new_student = [[age, usage, sleep, gender_encoded]]
    pred_addiction = rf_addiction.predict(new_student)[0]

    # ----------- Modelo 2 ---------
    new_mental = [new_student[0] + [pred_addiction]]
    pred_mental = rf_mental.predict(new_mental)[0]

    # ----------- Modelo 3 ---------
    risk_level = rf_addiction_level.predict(new_mental)[0]

    # ----------- Modelo 4 ---------
    acad_risk = rf_risk.predict(new_mental)[0]

    return f"""
    <h2>Resultado de Predicción</h2>
    <p>Score de Adicción: {pred_addiction:.2f}/10</p>
    <p>Score de Salud Mental: {pred_mental:.2f}/10</p>
    <p>Nivel de Riesgo de Adicción: {risk_level}</p>
    <p>Riesgo Académico Final: {acad_risk}</p>
    <br>
    <a href="/">Volver</a>
    """
    

if __name__ == "__main__":
    app.run(debug=True)
