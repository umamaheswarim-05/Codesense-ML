"""
CodeSense - Random Forest Model Training Script
Trains a model to predict student risk level based on their submission history.
Run this script whenever you want to retrain the model with latest data.
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    return psycopg2.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
    )

def build_dataset():
    conn = get_connection()

    query = """
        SELECT
            u.user_id,
            u.first_name || ' ' || u.last_name AS name,
            COUNT(e.exec_id) AS total_submissions,
            COUNT(e.exec_id) FILTER (WHERE e.is_success = true) AS successful_runs,
            COUNT(e.exec_id) FILTER (WHERE e.is_success = false) AS failed_runs,
            COUNT(DISTINCT er.error_type) AS distinct_error_types,
            MODE() WITHIN GROUP (ORDER BY er.error_type) AS common_error
        FROM users u
        LEFT JOIN executions e ON u.user_id = e.user_id
        LEFT JOIN errors er ON e.exec_id = er.exec_id
        WHERE u.role = 'student'
        GROUP BY u.user_id, u.first_name, u.last_name
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df = df.fillna(0)
    df["total_submissions"] = df["total_submissions"].astype(int)
    df["successful_runs"] = df["successful_runs"].astype(int)
    df["failed_runs"] = df["failed_runs"].astype(int)

    df["error_rate"] = df.apply(
        lambda row: row["failed_runs"] / row["total_submissions"] if row["total_submissions"] > 0 else 0,
        axis=1
    )

    return df


def generate_synthetic_training_data(n=200):
    np.random.seed(42)
    rows = []

    for _ in range(n):
        total = np.random.randint(5, 150)
        error_rate = np.random.uniform(0, 1)
        failed = int(total * error_rate)
        successful = total - failed
        distinct_errors = np.random.randint(0, 4)

        if error_rate >= 0.55:
            risk = "High"
        elif error_rate >= 0.28:
            risk = "Medium"
        else:
            risk = "Low"

        rows.append({
            "total_submissions": total,
            "successful_runs": successful,
            "failed_runs": failed,
            "distinct_error_types": distinct_errors,
            "error_rate": error_rate,
            "risk_level": risk
        })

    return pd.DataFrame(rows)


def train_model():
    print("Generating training dataset...")
    df = generate_synthetic_training_data(300)

    features = ["total_submissions", "successful_runs", "failed_runs", "distinct_error_types", "error_rate"]
    X = df[features]
    y = df["risk_level"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model trained. Test accuracy: {accuracy:.2%}")

    joblib.dump(model, "risk_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("Model saved as risk_model.pkl")

    return model, le


def predict_and_save():
    print("\nFetching real student data from database...")
    df = build_dataset()

    if df.empty:
        print("No students found in database.")
        return

    model = joblib.load("risk_model.pkl")
    le = joblib.load("label_encoder.pkl")

    features = ["total_submissions", "successful_runs", "failed_runs", "distinct_error_types", "error_rate"]

    conn = get_connection()
    cur = conn.cursor()

    for _, row in df.iterrows():
        if row["total_submissions"] == 0:
            continue

        X_student = pd.DataFrame([row[features]])
        prediction_encoded = model.predict(X_student)[0]
        risk_level = le.inverse_transform([prediction_encoded])[0]

        proba = model.predict_proba(X_student)[0]
        risk_classes = le.classes_
        weight_map = {"Low": 20, "Medium": 55, "High": 90}
        risk_score = int(sum(proba[i] * weight_map[risk_classes[i]] for i in range(len(risk_classes))))

        common_error = row["common_error"] if row["common_error"] else None

        cur.execute("""
            DELETE FROM ml_predictions WHERE user_id = %s
        """, (int(row["user_id"]),))

        cur.execute("""
            INSERT INTO ml_predictions (user_id, risk_score, risk_level, common_error)
            VALUES (%s, %s, %s, %s)
        """, (int(row["user_id"]), risk_score, risk_level, common_error))

        print(f"  {row['name']}: Risk={risk_level} ({risk_score}%) | Submissions={row['total_submissions']} | ErrorRate={row['error_rate']:.0%}")

    conn.commit()
    cur.close()
    conn.close()
    print("\nPredictions saved to ml_predictions table.")


if __name__ == "__main__":
    train_model()
    predict_and_save()
    print("\nDone! Run this script again anytime to refresh predictions with latest data.")