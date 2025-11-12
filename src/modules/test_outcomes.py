# Create test_outcomes.py
import sqlite3

conn = sqlite3.connect('src/modules/data/learning.db')
cursor = conn.cursor()

# Check how many predictions have outcomes
cursor.execute("SELECT COUNT(*) FROM predictions")
total_preds = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM outcomes")
total_outcomes = cursor.fetchone()[0]

cursor.execute("SELECT COUNT(*) FROM outcomes WHERE success = 0")
wrong_outcomes = cursor.fetchone()[0]

print(f"Total predictions: {total_preds}")
print(f"Predictions with outcomes: {total_outcomes}")
print(f"Wrong predictions: {wrong_outcomes}")

conn.close()
