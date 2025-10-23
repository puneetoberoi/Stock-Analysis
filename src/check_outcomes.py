#!/usr/bin/env python3
"""
Evening outcome checker - runs standalone
Checks how predictions performed and updates learning system
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from your main file
from src.main import check_prediction_outcomes

async def main():
    results = await check_prediction_outcomes()
    print(f"Outcome check complete: {results}")

if __name__ == "__main__":
    asyncio.run(main())
