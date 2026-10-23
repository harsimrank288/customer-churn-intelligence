import json
import sqlite3
import database


def test_save_prediction(tmp_path, monkeypatch):
    # Create a temporary database for this test
    test_db = tmp_path / "test_churn.db"

    # Redirect database connections to the temporary database
    def get_test_connection():
        connection = sqlite3.connect(test_db)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(
    database,
    "get_connection",
    get_test_connection
)

    # Create a customer record
    customer_data = {
        "gender": "Female",
        "tenure": 12,
        "Contract": "Month-to-month",
        "MonthlyCharges": 85.5
    }

    # Create the table
    connection = database.get_connection()

    try:
        connection.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_data TEXT NOT NULL,
                predicted_churn TEXT NOT NULL,
                churn_probability REAL NOT NULL,
                risk_level TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.commit()
    finally:
        connection.close()

    # Save the prediction
    prediction_id = database.save_prediction(
        customer_data=json.dumps(customer_data),
        predicted_churn="Yes",
        churn_probability=88.36,
        risk_level="High"
    )

    # Verify that the prediction was saved
    connection = database.get_connection()

    try:
        record = connection.execute(
            "SELECT * FROM predictions WHERE id = ?",
            (prediction_id,)
        ).fetchone()

        assert record is not None
        assert record["predicted_churn"] == "Yes"
        assert record["churn_probability"] == 88.36
        assert record["risk_level"] == "High"

        saved_customer = json.loads(record["customer_data"])
        assert saved_customer["tenure"] == 12

    finally:
        connection.close()