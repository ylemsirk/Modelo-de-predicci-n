from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

app = Flask(__name__)

df = pd.read_csv('Students Social Media Addiction_enriched.csv')
df = df.drop_duplicates().dropna(how="all")

X_ADDITC = df[['Age','Gender','Avg_Daily_Usage_Hours','Sleep_Hours_Per_Night']].copy()
y_adiccion = df['Addicted_Score']

X_ENCODED_ADDICT= pd.get_dummies(X_ADDITC, columns=["Gender"], drop_first=True)

X_train_addict, X_test_addict, y_train_addict, y_test_addict = train_test_split(
    X_ENCODED_ADDICT, y_adiccion, test_size=0.2, random_state=42
)

rf_addiction = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42)
rf_addiction.fit(X_train_addict, y_train_addict)

y_salud_mental = df['Mental_Health_Score']
X_ENCODED_MENTAL = X_ENCODED_ADDICT.copy()
X_ENCODED_MENTAL["Addicted_Score"] = df['Addicted_Score']

X_train_mental, X_test_mental, y_train_mental, y_test_mental = train_test_split(
    X_ENCODED_MENTAL, y_salud_mental, test_size=0.2, random_state=42
)

rf_mental = RandomForestRegressor(n_estimators=100, max_features="sqrt", random_state=42)
rf_mental.fit(X_train_mental, y_train_mental)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/resultado', methods=['POST'])
def calcular():
    age = float(request.form['age'])
    usage = float(request.form['usage'])
    sleep = float(request.form['sleep'])
    gender = request.form['gender']

    gender_encoded = 1 if gender == "male" else 0

    new_student = [[age, usage, sleep, gender_encoded]]
    pred_addiction = rf_addiction.predict(new_student)[0]

    new_mental = [new_student[0] + [pred_addiction]]
    pred_mental = rf_mental.predict(new_mental)[0]

    return render_template(
        'resultado.html', 
        adiccion=pred_addiction, 
        mental=pred_mental
    )

if __name__ == '__main__':
    app.run()
