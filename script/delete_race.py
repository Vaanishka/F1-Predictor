import sqlite3
from pathlib import Path

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE = BASE_DIR / "database" / "f1_predictions.db"

RACE_NAME = "Singapore Grand Prix"        
YEAR = 2024
# ---------------------------------------


def delete_race_data(db_path, race_name, year):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Get race_ids first (explicit + safe)
        cursor.execute("""
            SELECT race_id
            FROM races
            WHERE race_name = ? AND year = ?
        """, (race_name, year))

        race_ids = [row[0] for row in cursor.fetchall()]

        if not race_ids:
            print("No matching race found. Nothing to delete.")
            return

        print(f"Deleting data for race_id(s): {race_ids}")

        # Delete dependent tables first
        cursor.executemany(
            "DELETE FROM race_features WHERE race_id = ?",
            [(rid,) for rid in race_ids]
        )

        cursor.executemany(
            "DELETE FROM predictions WHERE race_id = ?",
            [(rid,) for rid in race_ids]
        )

        # Delete race entries
        cursor.executemany(
            "DELETE FROM races WHERE race_id = ?",
            [(rid,) for rid in race_ids]
        )

        conn.commit()
        print("Deletion completed successfully.")

    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Database deletion failed: {e}")

    finally:
        conn.close()


if __name__ == "__main__":
    delete_race_data(DB_FILE, RACE_NAME, YEAR)
