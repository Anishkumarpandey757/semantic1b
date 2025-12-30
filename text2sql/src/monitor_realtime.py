#!/usr/bin/env python3
"""
Real-time training monitor - Updates every 10 seconds
Shows actual progress from checkpoint files
"""
import json
import time
import os
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_latest_training():
    """Find latest training directory"""
    training_dirs = sorted(CHECKPOINT_DIR.glob("qwen_cpu_text2sql_*"))
    return training_dirs[-1] if training_dirs else None

def get_progress(training_dir):
    """Extract progress from latest checkpoint"""
    checkpoints = sorted(training_dir.glob("checkpoint-*"))
    
    if not checkpoints:
        return None
    
    latest_cp = checkpoints[-1]
    state_file = latest_cp / "trainer_state.json"
    
    if not state_file.exists():
        return None
    
    with open(state_file) as f:
        state = json.load(f)
    
    return {
        'step': state.get('global_step', 0),
        'max_steps': state.get('max_steps', 500),
        'epoch': state.get('epoch', 0),
        'log_history': state.get('log_history', [])
    }

def display_progress(progress, training_dir):
    """Display formatted progress"""
    clear_screen()
    print("=" * 70)
    print("🔄 REAL-TIME TRAINING MONITOR")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Training Dir: {training_dir.name}")
    print("=" * 70)
    
    if progress is None:
        print("\n⏳ Waiting for first checkpoint...")
        print("   Training is initializing...")
        print("   First checkpoint will appear at step 100")
        print("\n💡 This is normal - CPU training is VERY slow")
        print("   Each step takes 30-60 seconds on CPU")
        print("   Estimated time for step 100: 50-100 minutes")
    else:
        step = progress['step']
        max_steps = progress['max_steps']
        percent = (step / max_steps) * 100
        
        print(f"\n📊 Progress: Step {step}/{max_steps} ({percent:.1f}%)")
        print(f"📈 Epoch: {progress['epoch']:.2f}")
        
        # Progress bar
        bar_length = 50
        filled = int(bar_length * step / max_steps)
        bar = '█' * filled + '░' * (bar_length - filled)
        print(f"\n[{bar}]")
        
        # Latest metrics
        if progress['log_history']:
            latest = progress['log_history'][-1]
            print(f"\n📉 Latest Metrics:")
            if 'loss' in latest:
                print(f"   Training Loss: {latest['loss']:.4f}")
            if 'eval_loss' in latest:
                print(f"   Eval Loss: {latest['eval_loss']:.4f}")
            if 'learning_rate' in latest:
                print(f"   Learning Rate: {latest['learning_rate']:.2e}")
            if 'epoch' in latest:
                print(f"   Epoch: {latest['epoch']:.2f}")
        
        # Time estimate
        remaining = max_steps - step
        print(f"\n⏱️  Remaining Steps: {remaining}")
        print(f"   Estimated time: {remaining * 0.8:.0f}-{remaining * 1.5:.0f} minutes")
    
    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop monitoring (training continues)")
    print("=" * 70)

def main():
    """Main monitoring loop"""
    print("Starting real-time monitor...")
    print("Checking for training progress every 10 seconds...")
    time.sleep(2)
    
    try:
        while True:
            training_dir = get_latest_training()
            
            if training_dir is None:
                clear_screen()
                print("=" * 70)
                print("❌ NO TRAINING FOUND")
                print("=" * 70)
                print("\nNo training directory detected.")
                print("Make sure training is running first.")
                break
            
            progress = get_progress(training_dir)
            display_progress(progress, training_dir)
            
            # Check if complete
            if progress and progress['step'] >= progress['max_steps']:
                print("\n✅ TRAINING COMPLETE!")
                break
            
            time.sleep(10)  # Update every 10 seconds
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped (training still running in background)")

if __name__ == "__main__":
    main()
