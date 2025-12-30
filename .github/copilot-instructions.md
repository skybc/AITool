# AI Copilot Instructions for AITool

## Project Overview
**AITool** is a PyQt6-based industrial defect detection system using YOLO11. It provides complete training, inference, and real-time visualization capabilities with a modular architecture supporting multiple consumers for flexible data processing.

**Key Technologies:**
- YOLO11 (ultralytics) - Object detection backbone
- PySide6 - GUI framework
- PyTorch - Deep learning framework
- Python 3.11+ with conda environment

---

## Critical Architecture Patterns

### 1. **Output Consumer Pattern** (Custom Pub-Sub Architecture)
**Location:** `core/output_consumer.py`, `gui/training_progress_consumer.py`

This is THE core design pattern for training output handling. YOLO outputs go through a multi-layer pipeline:

```
YOLO Training → TeeWriter (splits output)
  ├→ StringIO buffer (in-memory)
  ├→ Log file (rotating, 5MB/file)
  └→ Console

StringIO → OutputConsumerManager
  ├→ TrainingProgressConsumer (parses metrics)
  └→ Custom consumers (extend OutputConsumer base class)
```

**Why this pattern:** Decouples YOLO output parsing from UI updates. Multiple consumers can independently process the same output stream.

**Implementation pattern:**
```python
from core.output_consumer import OutputConsumer

class MyConsumer(OutputConsumer):
    def consume(self, output: str) -> None:
        # Parse output incrementally
        pass
    
    def on_training_start(self) -> None:
        # Optional: called when training begins
        pass
```

**Register in training thread:** `TrainThread.register_output_consumer(consumer)`

### 2. **Training Thread Model**
**Location:** `gui/train_widget.py` - `TrainThread` class

Training runs on a separate QThread to prevent GUI freezing. Key aspects:
- Output captured through `TeeWriter` (custom context manager)
- Consumers notified via `log_signal` Qt signal
- Training state tracked via `is_training` flag
- Proper cleanup on `finished_signal` emission

**Critical workflow:**
1. `TrainWidget.start_training()` → spawns `TrainThread`
2. `TrainThread.run()` → executes YOLO training with consumer notifications
3. `TrainThread` → emits `finished_signal` on completion
4. `TrainWidget.on_training_finished()` → unregisters consumers, updates UI

### 3. **Model Management Flow**
**Location:** `core/train.py` - `DefectDetector._init_model()`

Models are organized in a specific directory structure:
- **Source:** `yolopt/11/yolo11{n|s|m|l|x}.pt` (pre-trained, downloaded once)
- **Output:** `results/yolo11_defect/weights/{best.pt, last.pt}`
- **Quick access:** `data/models/` (symlinks/copies)

**Important:** `download_models.py` must run before first training. Models are ~500MB total.

---

## Developer Workflows

### Setup & Running
```bash
# Environment setup (one-time)
conda create -n defect_detection python=3.11
conda activate defect_detection
pip install -r requirements.txt

# Download YOLO11 models (one-time)
python download_models.py

# Start GUI
python app.py

# View logs (while training)
tail -f results/train.log
```

### Build Executable
**Files:** `build_single_exe.py`, `build_exe.py`

Uses PyInstaller. May fail due to:
1. Missing `yolopt/11/` models → run `download_models.py` first
2. OpenMP conflicts → handled via env vars in `app.py`
3. Qt plugin issues → check `build_temp/AITool/qt.conf`

### Testing & Debugging
- GUI: `python app.py` - direct execution, no terminal isolation
- Training: Set small `epochs: 1` in `config.yaml` for quick validation
- Consumer parsing: Check `TrainingProgressConsumer` regex patterns for ANSI code handling

---

## Project-Specific Conventions

### File Organization
| Folder | Purpose | Key Files |
|--------|---------|-----------|
| `core/` | Training logic, output handling | `train.py`, `output_consumer.py`, `output_logger.py` |
| `gui/` | UI components, thread management | `main_window.py`, `train_widget.py`, `detect_widget.py` |
| `yolopt/11/` | Pre-trained YOLO11 models | Auto-populated by `download_models.py` |
| `results/` | Training outputs | `train.log` (rotating), `yolo11_defect/weights/` |
| `data/` | Dataset + model copies | `images/{train,val,test}`, `labels/`, `models/` |

### Configuration
- **Training config:** `config.yaml` (YAML format, defines model, epochs, batch size, etc.)
- **Dataset config:** `data/data.yaml` (path, train/val splits, class names)
- **Access pattern:** `yaml.safe_load(open(path))` - see `train.py:DefectDetector.__init__`

### Logging
- **System:** `core/output_logger.py` implements rotating file handler (5MB threshold, 100 backups)
- **Output:** Every training generates `results/train_log_YYYYMMDD_HHMMSS.txt`
- **Current training:** Tailed via `results/train.log` symlink

### GUI Thread Safety
- All Qt signals used for thread communication (never direct object access across threads)
- `TrainWidget` uses `log_signal` to append console output
- Training state: `is_training` boolean flag (checked in close event)

---

## Integration Points & Data Flows

### Dataset Validation
**Location:** `gui/train_widget.py:TrainThread.run()` - Lines ~70-100

Checks:
- `data.yaml` exists and is parseable
- `images/train` and `images/val` directories exist
- At least 1 image in each directory (jpg/png)

YOLO auto-validates label format; custom validation happens before training starts.

### YOLO Output Parsing
**Location:** `gui/training_progress_consumer.py`

Parses lines like:
```
\x1b[K        1/100       0G      2.325      10.21       1.62       1106        640: 0%
```
Extracts: epoch, GPU memory, box_loss, cls_loss, dfl_loss, instances, image_size

Handles ANSI codes (`\x1b[K`), `\r` vs `\n` line endings, leading whitespace.

### External Dependencies
- **ultralytics/YOLO:** Auto-downloads from Hugging Face if model file missing
- **PyTorch/CUDA:** Detected automatically; CPU fallback in `DefectDetector.__init__`
- **Qt:** PySide6 version must match binary compatibility (6.5.0+)

---

## Common Patterns & Code Examples

### Adding a New Consumer
```python
# File: gui/custom_consumer.py
from core.output_consumer import OutputConsumer

class MetricsExporter(OutputConsumer):
    def __init__(self):
        self.metrics = []
    
    def consume(self, output: str) -> None:
        # Parse training metrics from output
        if "Epoch" in output:
            # Extract and store
            self.metrics.append(...)
    
    def on_training_end(self) -> None:
        # Export to JSON, CSV, etc.
        pass

# In TrainWidget.start_training():
exporter = MetricsExporter()
self.train_thread.register_output_consumer(exporter)
```

### Accessing Training Results Programmatically
```python
from pathlib import Path
import yaml

# Find best model
best_model = Path('results/yolo11_defect/weights/best.pt')

# Read training config
with open('config.yaml') as f:
    config = yaml.safe_load(f)
epochs = config['training']['epochs']
```

### Modifying Logging Behavior
Edit `core/output_logger.py:OutputLogger.setup_logger()`:
```python
MAX_FILE_SIZE = 10 * 1024 * 1024  # Change from 5MB
BACKUP_COUNT = 50  # Fewer backups
```

---

## Known Issues & Workarounds

| Issue | Root Cause | Solution |
|-------|-----------|----------|
| Models not found | `download_models.py` not run | Run once: `python download_models.py` |
| GUI freezes during training | Output processing on main thread | All processing happens in `TrainThread` |
| "YOLO weights not found" error | Missing `yolopt/11/` directory | Ensure download_models.py completed |
| OpenMP conflicts on macOS/Linux | MKL threading | Already set: `KMP_DUPLICATE_LIB_OK=TRUE` in `app.py` |
| Slow training on GPU | Wrong device selection | Check `config.yaml:training:device` - should be `cuda:0` if GPU available |

---

## Key Files Reference

| File | Purpose | ~Lines |
|------|---------|--------|
| `app.py` | Entry point, env var setup | 30 |
| `core/train.py` | YOLO trainer wrapper, model loading | 313 |
| `core/output_consumer.py` | Consumer base class, manager | 107 |
| `core/output_logger.py` | Rotating file logger, TeeWriter | 150 |
| `gui/main_window.py` | Main window, tab switching | 110 |
| `gui/train_widget.py` | Training UI, thread management | 748 |
| `gui/training_progress_consumer.py` | Progress parsing implementation | ~100 |
| `config.yaml` | Training hyperparameters | 35 lines |
| `download_models.py` | YOLO11 model download utility | ~50 |

---

## Important Reminders for AI Agents

1. **Always check for models first** - Many training failures stem from missing `yolopt/11/` directory
2. **Thread-safe UI updates** - Use Qt signals, never call `QWidget.setText()` from `QThread`
3. **Consumer registration timing** - Register BEFORE training starts, unregister AFTER it finishes
4. **Config overrides** - `config.yaml` + `data.yaml` control everything; changes require app restart or manual trainer reinit
5. **Dataset paths must be absolute** - Relative paths in `data.yaml` cause YOLO to fail silently
6. **Output parsing is stateful** - `TrainingProgressConsumer` maintains line buffer across `consume()` calls; don't reset it
