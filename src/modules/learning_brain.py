"""
Learning Brain Module - Simplified version
"""

import sqlite3
import json
from datetime import datetime, timedelta
import os

class LearningBrain:
    def __init__(self, db_path=None):
        """Initialize the learning brain with database"""
        if db_path is None:
            # FIXED: Create path relative to THIS FILE's location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(current_dir, 'data')
            db_path = os.path.join(data_dir, 'learning.db')
        
        self.db_path = db_path
        print(f"🔍 Database will be created at: {self.db_path}")  # Debug line
        self.init_database()
        
    def init_database(self):
        """Create database and tables if they don't exist"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Connect and create tables
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables directly (no external schema file)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                stock TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence FLOAT,
                target_date DATE,
                entry_price FLOAT,
                llm_model TEXT,
                reasoning TEXT,
                rsi FLOAT,
                macd FLOAT,
                volume_ratio FLOAT,
                patterns TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL,
                check_date DATE DEFAULT CURRENT_DATE,
                actual_price FLOAT,
                actual_move_pct FLOAT,
                success BOOLEAN,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accuracy_tracking (
                stock TEXT,
                llm_model TEXT,
                total_predictions INTEGER DEFAULT 0,
                correct_predictions INTEGER DEFAULT 0,
                accuracy_pct FLOAT DEFAULT 0,
                last_updated DATE DEFAULT CURRENT_DATE,
                PRIMARY KEY(stock, llm_model)
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"✅ Database initialized at {self.db_path}")
        
    def record_prediction(self, stock, prediction, confidence, price, 
                         llm_model, reasoning, indicators=None):
        """Record a new prediction"""
        if indicators is None:
            indicators = {}
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate target date (1 week from now)
        target_date = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        
        cursor.execute("""
            INSERT INTO predictions 
            (stock, prediction, confidence, target_date, entry_price, 
             llm_model, reasoning, rsi, macd, volume_ratio, patterns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            stock, 
            prediction, 
            confidence,
            target_date,
            price,
            llm_model,
            reasoning,
            indicators.get('rsi', 0),
            indicators.get('macd', 0),
            indicators.get('volume_ratio', 0),
            indicators.get('patterns', '')
        ))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        print(f"📝 Recorded prediction #{prediction_id}: {stock} -> {prediction} (confidence: {confidence:.1f}%)")
        return prediction_id
        
    def get_accuracy_report(self):
        """Generate accuracy report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get prediction count
        cursor.execute("SELECT COUNT(*) FROM predictions")
        pred_count = cursor.fetchone()[0]
        
        conn.close()
        
        if pred_count == 0:
            return "No prediction history yet"
        
        report = f"\n📊 **ACCURACY REPORT**\n"
        report += f"Total predictions recorded: {pred_count}\n"
        
        return report


# Simple test
if __name__ == "__main__":
    print("Starting Learning Brain test...")
    
    # Create instance
    brain = LearningBrain()
    
    # Test recording
    brain.record_prediction(
        stock='AAPL',
        prediction='BUY',
        confidence=75.5,
        price=175.50,
        llm_model='test',
        reasoning='Test prediction'
    )
    
    # Show report
    print(brain.get_accuracy_report())
    print("✅ Test completed!")
