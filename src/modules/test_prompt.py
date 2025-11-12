# test_prompt.py

# Add the 'src/modules' folder to the path so we can import our learner
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src/modules'))

from autonomous_learner import AutonomousLearner

# --- This is what your main.py will do ---
print("--- Simulating main.py getting a prompt for the LLM ---")

# 1. Create a learner instance
learner = AutonomousLearner()

# 2. Get the latest learnings
learning_context = learner.get_learning_prompt()

# 3. Print the prompt that will be sent to Groq
print("\n✅ LEARNINGS LOADED. THE LLM WILL SEE THIS:\n")
print("-----------------------------------------")
print(learning_context)
print("Analyze AAPL and provide BUY/SELL/HOLD...")
print("-----------------------------------------")
