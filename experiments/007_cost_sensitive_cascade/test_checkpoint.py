"""Test checkpoint system for Experiment 007."""
import sys
from pathlib import Path
import tempfile
import torch

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from run import CheckpointManager, CostSensitiveLoss


def test_checkpoint_manager():
    """Test CheckpointManager functionality."""
    print("Testing CheckpointManager...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {'checkpoint': {'checkpoint_file': 'test.pt'}}
        cm = CheckpointManager(tmpdir, config)
        
        # Test save
        model_state = {'weight': torch.randn(10, 10)}
        optimizer_state = {'lr': 0.001}
        metrics = {'accuracy': 0.85}
        best_metrics = {'best_accuracy': 0.90}
        
        cm.save_checkpoint(
            stage='stage1',
            epoch=5,
            model_state=model_state,
            optimizer_state=optimizer_state,
            scheduler_state=None,
            metrics=metrics,
            best_metrics=best_metrics,
            is_best=True
        )
        
        # Test load
        ckpt = cm.load_checkpoint()
        assert ckpt is not None, "Checkpoint load failed"
        assert ckpt['stage'] == 'stage1', "Stage mismatch"
        assert ckpt['epoch'] == 5, "Epoch mismatch"
        
        print("  [OK] Save checkpoint passed")
        print("  [OK] Load checkpoint passed")
    
    print("[OK] CheckpointManager test passed\n")


def test_cost_sensitive_loss():
    """Test CostSensitiveLoss computation."""
    print("Testing CostSensitiveLoss...")
    
    # Create cost matrix (same as config)
    cost_matrix = torch.tensor([
        [1.0, 2.0, 5.0, 10.0],
        [2.0, 1.0, 3.0, 8.0],
        [5.0, 3.0, 1.0, 5.0],
        [20.0, 15.0, 5.0, 1.0]
    ])
    
    criterion = CostSensitiveLoss(cost_matrix)
    
    # Create dummy inputs
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 4, (8,))
    
    # Compute loss
    loss = criterion(logits, labels)
    
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    
    print(f"  [OK] Loss computed: {loss.item():.4f}")
    print("[OK] CostSensitiveLoss test passed\n")


def test_resume_simulation():
    """Simulate interrupt and resume."""
    print("Testing interrupt/resume simulation...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {'checkpoint': {'checkpoint_file': 'test.pt'}}
        cm = CheckpointManager(tmpdir, config)
        
        # Simulate training epoch 3
        model_state = {'epoch_3_weight': torch.randn(5, 5)}
        cm.save_checkpoint(
            stage='stage1',
            epoch=3,
            model_state=model_state,
            optimizer_state={'step': 300},
            scheduler_state=None,
            metrics={'recall': 0.75},
            best_metrics={'best_recall': 0.80},
        )
        
        # Simulate interrupt and resume
        ckpt = cm.load_checkpoint()
        assert ckpt['epoch'] == 3, "Should resume from epoch 3"
        assert 'epoch_3_weight' in ckpt['model_state_dict'], "Model state should be preserved"
        
        print("  [OK] Interrupt at epoch 3")
        print("  [OK] Resume from epoch 3")
        print("  [OK] State preserved correctly")
    
    print("[OK] Resume simulation test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Experiment 007 Checkpoint System Tests")
    print("=" * 60 + "\n")
    
    test_checkpoint_manager()
    test_cost_sensitive_loss()
    test_resume_simulation()
    
    print("=" * 60)
    print("All tests passed! Checkpoint system is working correctly.")
    print("You can now run: python run.py")
    print("If interrupted, run again to auto-resume.")
    print("=" * 60)
