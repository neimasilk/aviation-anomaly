"""
Safe runner for Experiment 006 with file logging.
This avoids PowerShell Tee-Object buffer overflow issues.
"""
import sys
import logging
from pathlib import Path
from datetime import datetime

def setup_logging(log_dir: Path):
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"006_training_{timestamp}.log"
    
    # Setup root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def main():
    # Setup logging first
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_file = setup_logging(log_dir)
    
    logging.info("="*60)
    logging.info("Experiment 006: SMOTE-Augmented Training (FIXED)")
    logging.info("="*60)
    logging.info(f"Log file: {log_file}")
    logging.info(f"Python: {sys.version}")
    logging.info(f"Platform: {sys.platform}")
    
    try:
        # Import and run the actual experiment
        import run
        
        exp_dir = Path(__file__).parent
        runner = run.ExperimentRunner(exp_dir)
        runner.run()
        
        logging.info("="*60)
        logging.info("Experiment completed successfully!")
        logging.info("="*60)
        
    except Exception as e:
        logging.error("="*60)
        logging.error(f"Experiment failed with error: {e}")
        logging.error("="*60)
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
