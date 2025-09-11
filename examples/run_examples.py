#!/usr/bin/env python3
"""
Example scripts showing how to use the finetuning system with different datasets
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the result"""
    print(f"\n{=*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{=*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False
    return True

def main():
    print("Historical Text Dating - Finetuning Examples")
    print("=" * 50)
    
    # Test with sample data (should always work)
    print("\n1. Testing with sample data...")
    success = run_command(
        "python test_setup.py",
        "Test with sample data"
    )
    
    if not success:
        print("❌ Basic test failed. Please check your setup.")
        return
    
    # Test with Ben Yehuda dataset
    print("\n2. Testing with Ben Yehuda dataset...")
    run_command(
        "python src/main.py data=ben_yehuda",
        "Train with Ben Yehuda dataset"
    )
    
    # Test with Sefaria dataset
    print("\n3. Testing with Sefaria dataset...")
    run_command(
        "python src/main.py data=sefaria",
        "Train with Sefaria dataset"
    )
    
    # Test with custom parameters
    print("\n4. Testing with custom parameters...")
    run_command(
        "python src/main.py data=ben_yehuda training.batch_size=4 training.learning_rate=1e-5",
        "Train with custom parameters"
    )
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check the outputs/ directory for training results")
    print("Check WandB for experiment tracking (if configured)")

if __name__ == "__main__":
    main()
