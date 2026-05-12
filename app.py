from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

app = Flask(__name__)
app.secret_key = "teen_mental_health_secret"

#  LOAD & PREPROCESS DATA 
df = pd.read_csv("Teen_Mental_Health_Dataset.csv")
df.dropna(inplace=True)

# Replace LabelEncoder with One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

X = df.drop("depression_label", axis=1)
y = df["depression_label"]

# Split dataset with balanced classes
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store trained models
trained_models = {}

# ROUTES 

@app.route("/")
def index():
    columns = list(X.columns)
    return render_template("index.html", columns=columns)


@app.route("/train", methods=["POST"])
def train():

    data = request.json

    model_choice = data.get("model")
    optimizer_choice = data.get("optimizer", "adam")
    loss_choice = data.get("loss", "log_loss")

    if not model_choice:
        return jsonify({"error": "Please select a model"}), 400

    #  MODEL SELECTION 

    if model_choice == "Logistic Regression":

        model = LogisticRegression(max_iter=1000)

        model.fit(X_train_scaled, y_train)

        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=5
        )

        y_pred = model.predict(X_test_scaled)

    elif model_choice == "Decision Tree":

        model = DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5
        )

        y_pred = model.predict(X_test)

    elif model_choice == "Random Forest":

        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5
        )

        y_pred = model.predict(X_test)

    elif model_choice == "SVM":

        model = SVC()

        model.fit(X_train_scaled, y_train)

        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=5
        )

        y_pred = model.predict(X_test_scaled)

    elif model_choice == "Neural Network":

        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            solver=optimizer_choice,
            random_state=42
        )

        model.fit(X_train_scaled, y_train)

        scores = cross_val_score(
            model,
            X_train_scaled,
            y_train,
            cv=5
        )

        y_pred = model.predict(X_test_scaled)

    else:
        return jsonify({"error": "Unknown model"}), 400

    #  MODEL EVALUATION 

    acc = accuracy_score(y_test, y_pred)

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred).tolist()

    cv_accuracy = round(scores.mean() * 100, 2)

    # Save trained model
    trained_models[model_choice] = model

    return jsonify({
        "model": model_choice,
        "optimizer": optimizer_choice,
        "loss": loss_choice,
        "accuracy": round(acc * 100, 2),
        "cross_validation_accuracy": cv_accuracy,
        "confusion_matrix": cm,
        "report": report
    })


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json
    model_choice = data.get("model")

    if model_choice not in trained_models:
        return jsonify({"error": "Train the model first"}), 400

    model = trained_models[model_choice]

    # Select the correct sample type for each model
    if model_choice in ["Decision Tree", "Random Forest"]:
        sample = X_test.iloc[0].values.reshape(1, -1)
    else:
        sample = X_test_scaled[0].reshape(1, -1)

    prediction = int(model.predict(sample)[0])

    label = "Depressed" if prediction == 1 else "Not Depressed"

    feature_values = {
        col: float(X_test.iloc[0][col])
        for col in X.columns
    }

    return jsonify({
        "prediction": prediction,
        "label": label,
        "sample": feature_values
    })


@app.route("/dataset_info")
def dataset_info():

    info = {
        "rows": int(len(df)),
        "features": list(X.columns),
        "target": "depression_label",
        "class_dist": df["depression_label"].value_counts().to_dict(),
        "sample": df.head(5).to_dict(orient="records")
    }

    return jsonify(info)


if __name__ == "__main__":
    app.run(debug=True, port=5000)



