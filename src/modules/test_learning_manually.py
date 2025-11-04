"""
Simple test for Learning Brain module
"""

import os
import sys
from learning_brain import LearningBrain

def run_test():
    """Run a simple test of the learning brain"""
    
    print("="*60)
    print("🧪 LEARNING BRAIN TEST")
    print("="*60)
    
    # Step 1: Initialize
    print("\n📝 Step 1: Initializing Learning Brain...")
    brain = LearningBrain()
    
    # Step 2: Record test predictions
    print("\n📝 Step 2: Recording test predictions...")
    
    test_stocks = [
        ('AAPL', 'BUY', 85.0, 175.50, 'groq'),
        ('MSFT', 'HOLD', 70.0, 415.25, 'cohere'),
        ('GOOGL', 'SELL', 65.0, 155.80, 'gemini'),
    ]
    
    for stock, pred, conf, price, model in test_stocks:
        indicators = {
            'rsi': 65.5,
            'macd': 0.25,
            'volume_ratio': 1.2,
            'patterns': 'test_pattern'
        }
        
        pred_id = brain.record_prediction(
            stock=stock,
            prediction=pred,
            confidence=conf,
            price=price,
            llm_model=model,
            reasoning=f"Test prediction for {stock}",
            indicators=indicators
        )
        print(f"   ✓ Recorded: {stock} -> {pred} (ID: {pred_id})")
    
    # Step 3: Get report
    print("\n📊 Step 3: Generating report...")
    report = brain.get_accuracy_report()
    print(report)
    
    # Step 4: Verify database
    print("\n🔍 Step 4: Verifying database...")
    db_path = brain.db_path
    
    if os.path.exists(db_path):
        size = os.path.getsize(db_path)
        print(f"   ✅ Database exists: {db_path}")
        print(f"   📦 Size: {size} bytes")
    else:
        print(f"   ❌ Database NOT found at: {db_path}")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    return True

if __name__ == "__main__":
    try:
        success = run_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
