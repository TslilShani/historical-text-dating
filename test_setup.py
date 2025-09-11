#!/usr/bin/env python3
"""
Test script to verify the finetuning setup works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.main import main

if __name__ == "__main__":
    print("Testing the finetuning setup...")
    print("This will run with sample data if real data is not available.")
    print("To use real data, ensure the dataset paths in configs/data/ are correct.")
    print()
    
    try:
        main()
        print("✅ Test completed successfully!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
