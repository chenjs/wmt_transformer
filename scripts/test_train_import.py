#!/usr/bin/env python3
"""
Test imports and basic functionality of training components.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    print("✅ torch imported")

    from src.config import config
    print("✅ config imported")

    from src.data.tokenizer import load_tokenizers
    print("✅ load_tokenizers imported")

    from src.data.dataset import ParallelDataset
    print("✅ ParallelDataset imported")

    from src.model import Transformer
    print("✅ Transformer imported")

    from src.trainer import Trainer
    print("✅ Trainer imported")

    # Check tokenizer files
    data_dir = Path(__file__).parent.parent
    src_tokenizer_path = data_dir / "models_enhanced" / "src_tokenizer_final.model"
    tgt_tokenizer_path = data_dir / "src_tokenizer_final.model"

    if src_tokenizer_path.exists():
        print(f"✅ Source tokenizer exists: {src_tokenizer_path}")
    else:
        print(f"❌ Source tokenizer missing: {src_tokenizer_path}")

    if tgt_tokenizer_path.exists():
        print(f"✅ Target tokenizer exists: {tgt_tokenizer_path}")
    else:
        print(f"❌ Target tokenizer missing: {tgt_tokenizer_path}")

    # Check data files
    src_file = data_dir / "europarl-v7.de-en.en"
    tgt_file = data_dir / "europarl-v7.de-en.de"

    if src_file.exists():
        print(f"✅ Source data file exists: {src_file} ({src_file.stat().st_size:,} bytes)")
    else:
        print(f"❌ Source data file missing: {src_file}")

    if tgt_file.exists():
        print(f"✅ Target data file exists: {tgt_file} ({tgt_file.stat().st_size:,} bytes)")
    else:
        print(f"❌ Target data file missing: {tgt_file}")

    # Check config
    print(f"\nConfig check:")
    print(f"  max_len: {config.max_len}")
    print(f"  src_vocab_size: {config.src_vocab_size}")
    print(f"  tgt_vocab_size: {config.tgt_vocab_size}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  max_steps: {config.max_steps}")

    # Check for min_loss_improvement attribute
    if hasattr(config, 'min_loss_improvement'):
        print(f"  min_loss_improvement: {config.min_loss_improvement}")
    else:
        print(f"⚠️  min_loss_improvement not in config (will use default)")

    print("\n✅ All imports successful. Basic checks passed.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)