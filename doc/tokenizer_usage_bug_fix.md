# Tokenizer Usage Bug Fix - 2026-02-26

## Problem Description

The English-German translation transformer model was producing identical German outputs for all English inputs. This was caused by a critical bug in the training pipeline where German target text was incorrectly tokenized using the English tokenizer.

### Root Cause Analysis

**Bug**: The `create_batch` function in `src/data/batch.py` accepted only one tokenizer parameter but used it for both source (English) and target (German) text tokenization. During training, only the English tokenizer was passed, causing German text to be tokenized with incorrect vocabulary mappings.

**Impact**:
1. German text was tokenized with English vocabulary during training
2. Model learned to ignore target inputs
3. Evaluation used correct tokenizers, creating a training-inference mismatch
4. All inputs produced identical outputs

## Solution Overview

Fixed the tokenizer usage by modifying `create_batch` to accept separate source and target tokenizers, then updated all callers to pass both tokenizers correctly. Maintained backward compatibility for existing code.

## Changes Made

### 1. `src/data/batch.py` - Core batch creation logic

**Modified `create_batch` function**:
- Changed signature from `create_batch(samples, tokenizer, ...)` to `create_batch(samples, src_tokenizer, tgt_tokenizer=None, ...)`
- Added deprecation warning when `tgt_tokenizer` is not provided
- Updated tokenization logic to use `src_tokenizer` for source texts and `tgt_tokenizer` for target texts

**Modified `BatchIterator` class**:
- Updated `__init__` to accept `src_tokenizer` and `tgt_tokenizer` parameters
- Updated `__iter__` to pass both tokenizers to `create_batch`
- Added backward compatibility with deprecation warning

### 2. `src/trainer.py` - Training loop

**Modified `train_epoch` method**:
- Updated line 155 to pass both tokenizers: `create_batch(samples, self.src_tokenizer, self.tgt_tokenizer, ...)`

**Modified `Trainer.__init__`**:
- Computed target vocabulary size from `tgt_tokenizer.sp.get_piece_size()`
- Updated `LabelSmoothingLoss` to use `tgt_vocab_size` instead of `config.vocab_size`

### 3. `src/config.py` - Configuration structure

**Added separate vocabulary size fields**:
- Added `src_vocab_size: int = 32000`
- Added `tgt_vocab_size: int = 32000`
- Marked `vocab_size` as deprecated: `vocab_size: int = 32000  # Deprecated: use src_vocab_size and tgt_vocab_size`

### 4. `scripts/train.py` - Model creation and vocabulary handling

**Updated vocabulary size handling**:
- Set `config.src_vocab_size` and `config.tgt_vocab_size` from tokenizers
- Updated model creation to use separate vocabulary sizes:
  ```python
  model = Transformer(
      src_vocab_size=config.src_vocab_size,
      tgt_vocab_size=config.tgt_vocab_size,
      ...
  )
  ```

### 5. Inference Script Updates

**Updated `scripts/translate.py`, `debug_model.py`, `test_issue.py`**:
- Added logic to handle both old (single `vocab_size`) and new (separate `src_vocab_size`/`tgt_vocab_size`) checkpoint formats
- Compute vocabulary sizes from tokenizers for new models
- Use separate vocab sizes for model creation

## Backward Compatibility

All changes maintain backward compatibility:
1. `create_batch` accepts `tgt_tokenizer=None` and falls back to `src_tokenizer` with a deprecation warning
2. `BatchIterator` accepts `tgt_tokenizer=None` and falls back to `src_tokenizer` with a warning
3. Inference scripts check for both old and new config formats in checkpoints
4. Old checkpoints with single `vocab_size` field are automatically converted to use separate sizes

## Verification Process

### 1. Unit Tests
Created and ran `test_tokenizer_fix.py`:
- Verified `create_batch` uses separate tokenizers correctly
- Tested backward compatibility with single tokenizer
- Confirmed source and target texts are tokenized with correct tokenizers

### 2. Integration Tests
Created and ran `test_training_fix.py`:
- Simulated training with separate tokenizers having different vocab sizes
- Verified training loop runs without errors
- Confirmed `LabelSmoothingLoss` uses correct target vocabulary size

### 3. End-to-End Testing
Created and ran `test_full_training.py`:
- Ran actual training for 2 steps with the fixed code
- Verified training completes successfully
- Confirmed model parameters are created with correct vocabulary sizes

### 4. Existing Test Suite
Ran `test_issue.py`:
- Model still produces identical outputs due to previously trained weights
- However, the diagnostic script runs without errors
- New training will produce correct results

## Testing Results

All tests passed:
- ✅ Separate tokenizers test passed
- ✅ Backward compatibility test passed
- ✅ Training with separate tokenizers works
- ✅ `create_batch` works with separate tokenizers
- ✅ End-to-end training completed successfully

## Critical Notes

1. **Checkpoint Compatibility**: Existing checkpoints were trained with incorrect tokenization and may not be usable. Consider retraining from scratch for best results.

2. **Vocabulary Mismatch**: English and German tokenizers may have different vocabulary sizes. The model now correctly handles this difference.

3. **Performance Impact**: The fix ensures the model can properly learn translation patterns, which should significantly improve translation quality.

## Files Modified

1. `src/data/batch.py` - Lines 44-146
2. `src/trainer.py` - Lines 88-93, 158
3. `src/config.py` - Lines 16-18
4. `scripts/train.py` - Lines 64-67, 80-89
5. `scripts/translate.py` - Lines 50-99
6. `debug_model.py` - Lines 22-55
7. `test_issue.py` - Lines 26-58

## Next Steps

1. Retrain the model from scratch to leverage the fixed tokenization
2. Monitor training loss to ensure it decreases properly
3. Verify that different inputs produce different outputs after retraining
4. Consider updating any other scripts that might use `create_batch` or `BatchIterator`

## Conclusion

The tokenizer usage bug has been successfully fixed. The model now correctly tokenizes source and target texts with their respective tokenizers during training, eliminating the training-inference mismatch. This fix should enable the model to learn proper translation patterns and produce varied, accurate translations.