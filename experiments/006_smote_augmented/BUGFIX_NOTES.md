# Bug Fix Notes for Experiment 006

## Date: 2026-02-05

## Issues Found and Fixed

### 1. CRITICAL: Import Statement Inside Loop
**File:** `run.py` line 157  
**Problem:** `import pandas as pd` was inside `__getitem__` method, causing it to be executed thousands of times during training.

**Impact:**
- Severe performance degradation
- Memory fragmentation
- Potential reference counting issues leading to silent crashes

**Fix:** Moved import to top of file.

```python
# BEFORE (BAD)
def __getitem__(self, idx):
    import pandas as pd  # Executed every data access!
    sequence = [str(s) for s in sequence if pd.notna(s) and str(s).strip()]

# AFTER (GOOD)
import pandas as pd  # At top of file with other imports

def __getitem__(self, idx):
    sequence = [str(s) for s in sequence if pd.notna(s) and str(s).strip()]
```

### 2. CRITICAL: PowerShell Pipeline Buffer Overflow
**Problem:** Using `Tee-Object` with high-frequency output (tqdm progress bars) can cause buffer overflow and silent process termination.

**Symptom:** Training stops at random points (e.g., batch 604/734) without any error message.

**Fix Options:**
1. Use `run_safe.ps1` - PowerShell script with proper output redirection
2. Use `run_with_logging.py` - Python-native file logging
3. Run directly without Tee-Object: `python run.py > log.txt 2>&1`

### 3. MEDIUM: No CUDA OOM Handling
**Problem:** Training crashes without graceful handling when CUDA runs out of memory.

**Fix:** Added try-except blocks for `torch.cuda.OutOfMemoryError` and periodic cache clearing:
```python
if device == "cuda" and batch_idx % 100 == 0:
    torch.cuda.empty_cache()
```

### 4. LOW: torch.load Security Warning
**Problem:** `torch.load()` without `weights_only=True` can execute arbitrary code.

**Fix:** Added `weights_only=True` to all `torch.load()` calls.

```python
# BEFORE
torch.load(checkpoint_dir / "best_model.pt")

# AFTER
torch.load(checkpoint_dir / "best_model.pt", weights_only=True)
```

## Recommended Usage

### Option 1: Safe PowerShell Script (Recommended)
```powershell
cd experiments/006_smote_augmented
.\run_safe.ps1
```

### Option 2: Python Native Logging
```bash
cd experiments/006_smote_augmented
python run_with_logging.py
```

### Option 3: Direct with File Redirection
```bash
cd experiments/006_smote_augmented
python run.py 2>&1 | tee log.txt
```

## Avoid This (Causes Silent Crashes)
```powershell
# DON'T DO THIS - Tee-Object buffer overflow
python run.py 2>&1 | Tee-Object -FilePath log.txt
```

## Testing After Fix

Run a quick test with fewer epochs:
```python
# In config.yaml, temporarily set:
training:
  max_epochs: 2
```

Monitor GPU memory:
```bash
nvidia-smi -l 1  # Update every second
```

## Exit Codes Reference

| Exit Code | Meaning |
|-----------|---------|
| 0 | Success |
| -1073740791 | CUDA Out of Memory |
| -1073741819 | Access Violation (memory corruption) |
| 1 | General Python error |
| -1073741510 | CTRL+C / KeyboardInterrupt |

## Prevention Measures

1. **Never put imports inside loops or `__getitem__`**
2. **Use Python-native logging instead of shell pipelines for long-running tasks**
3. **Always handle CUDA OOM exceptions in GPU training**
4. **Use `weights_only=True` for `torch.load()`**
5. **Monitor memory usage during training**
