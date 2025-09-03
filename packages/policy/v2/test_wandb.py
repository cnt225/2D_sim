#!/usr/bin/env python3
"""
Test WandB connection
"""

import wandb
import time

print("Testing WandB connection...")

# Initialize wandb
try:
    run = wandb.init(
        project="tdot_rcfm_test",
        entity="cnt225-seoul-national-university",  # Your entity from login
        name="connection_test",
        config={"test": True}
    )
    
    print(f"✅ WandB connected successfully!")
    print(f"View run at: {run.url}")
    
    # Log some test data
    for i in range(3):
        wandb.log({"test_metric": i})
        time.sleep(1)
    
    wandb.finish()
    print("✅ Test complete - WandB is working!")
    
except Exception as e:
    print(f"❌ WandB connection failed: {e}")
    print("Please run: wandb login")