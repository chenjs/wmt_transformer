#!/usr/bin/env python3
"""
Validate that train.py data path fix is working correctly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import config to check values
from src.config import config

print("=" * 60)
print("Train.py Data Path Fix Validation")
print("=" * 60)

print("\n1. Checking config values:")
print(f"   config.src_file: {config.src_file}")
print(f"   config.tgt_file: {config.tgt_file}")
print(f"   config.src_tokenizer: {config.src_tokenizer}")
print(f"   config.tgt_tokenizer: {config.tgt_tokenizer}")
print(f"   config.min_loss_improvement: {getattr(config, 'min_loss_improvement', 'NOT SET')}")

# Check if cleaned data files exist
data_dir = Path(__file__).parent.parent
cleaned_src = data_dir / "models_enhanced" / "src_text_cleaned.txt"
cleaned_tgt = data_dir / "models_enhanced" / "tgt_text_cleaned.txt"
raw_src = data_dir / config.src_file
raw_tgt = data_dir / config.tgt_file

print("\n2. Checking data files:")
print(f"   Raw source data: {raw_src} - {'✅ EXISTS' if raw_src.exists() else '❌ MISSING'}")
if raw_src.exists():
    print(f"     Size: {raw_src.stat().st_size:,} bytes")
print(f"   Raw target data: {raw_tgt} - {'✅ EXISTS' if raw_tgt.exists() else '❌ MISSING'}")
if raw_tgt.exists():
    print(f"     Size: {raw_tgt.stat().st_size:,} bytes")
print(f"   Cleaned source data: {cleaned_src} - {'✅ EXISTS' if cleaned_src.exists() else '❌ MISSING'}")
if cleaned_src.exists():
    print(f"     Size: {cleaned_src.stat().st_size:,} bytes")
print(f"   Cleaned target data: {cleaned_tgt} - {'✅ EXISTS' if cleaned_tgt.exists() else '❌ MISSING'}")
if cleaned_tgt.exists():
    print(f"     Size: {cleaned_tgt.stat().st_size:,} bytes")

# Check tokenizer files
src_tokenizer_path = data_dir / config.src_tokenizer
tgt_tokenizer_path = data_dir / config.tgt_tokenizer

print("\n3. Checking tokenizer files:")
print(f"   Source tokenizer: {src_tokenizer_path} - {'✅ EXISTS' if src_tokenizer_path.exists() else '❌ MISSING'}")
print(f"   Target tokenizer: {tgt_tokenizer_path} - {'✅ EXISTS' if tgt_tokenizer_path.exists() else '❌ MISSING'}")

# Check train.py modifications
train_py_path = Path(__file__).parent / "train.py"
with open(train_py_path, 'r') as f:
    train_content = f.read()

print("\n4. Checking train.py modifications:")
if "config.src_file = \"models_enhanced/src_text_cleaned.txt\"" in train_content:
    print("   ✅ Data path override found in train.py")
else:
    print("   ❌ Data path override NOT found in train.py")

if "config.tgt_file = \"models_enhanced/tgt_text_cleaned.txt\"" in train_content:
    print("   ✅ Target data path override found in train.py")
else:
    print("   ❌ Target data path override NOT found in train.py")

# Check if we have the backup
backup_path = Path(__file__).parent / "train_backup.py"
if backup_path.exists():
    print(f"   ✅ Original train.py backed up at: {backup_path}")
else:
    print("   ⚠️  No backup of original train.py found")

print("\n5. Recommendation:")
if cleaned_src.exists() and cleaned_tgt.exists():
    print("   ✅ Cleaned data files exist - train.py should work correctly")
else:
    print("   ⚠️  Cleaned data files missing - run preprocess_enhanced.py first")

if src_tokenizer_path.exists() and tgt_tokenizer_path.exists():
    print("   ✅ Tokenizer files exist - training can proceed")
else:
    print("   ⚠️  Tokenizer files missing - run preprocess_enhanced.py first")

print("\n" + "=" * 60)
print("Validation complete!")
print("=" * 60)