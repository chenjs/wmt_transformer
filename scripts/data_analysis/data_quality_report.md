# Data Quality Analysis Report
**Date:** 2026-03-01
**Dataset:** Europarl v7 (English-German)
**Samples analyzed:** 10,000

## Key Findings

### 1. Sentence Length Distribution
- **Source (English):** Mean=25.6 words, Median=22.0, 90th percentile=47.0
- **Target (German):** Mean=23.2 words, Median=20.0, 90th percentile=43.0
- **Length ratio (src/tgt):** Mean=1.13, Median=1.10
- **Current max_len setting:** 32 tokens (may truncate ~10% of sentences at 90th percentile)

### 2. Vocabulary Usage
- **Source vocabulary:** 12,780 / 32,000 tokens used (40% coverage)
- **Target vocabulary:** 18,734 / 32,000 tokens used (58% coverage)
- **Source token distribution:**
  - Top 59 tokens cover 50% of occurrences
  - Top 2,232 tokens cover 90% of occurrences
  - Top 9,902 tokens cover 99% of occurrences
- **Target token distribution:**
  - Top 114 tokens cover 50% of occurrences
  - Top 5,017 tokens cover 90% of occurrences
  - Top 15,920 tokens cover 99% of occurrences

### 3. Data Quality Issues
- **Empty lines:** 49 (0.25%)
- **Very short lines (<3 words):** 234 (1.17%)
- **Very long lines (>100 words):** 36 (0.18%)
- **Non-printable characters:**
  - Source: 178 lines (1.8%) - likely encoding issues
  - **Target: 8,688 lines (86.9%) - CRITICAL: German umlauts (ä, ö, ü, ß) treated as non-printable**

### 4. Training Configuration
- **max_train_samples:** 200,000
- **Tokenizer training samples:** 100,000 (mismatch!)
- **batch_size:** 12
- **max_len:** 32

## Optimization Recommendations

### Immediate Actions (High Priority)
1. **Fix German character encoding:**
   - 86.9% of German lines contain "non-printable" characters
   - Likely due to umlauts (ä, ö, ü) and ß being incorrectly classified
   - Need to update character classification logic

2. **Increase tokenizer training samples:**
   - Tokenizer trained on 100k samples, but model uses 200k
   - Train tokenizer on full 200k samples for better coverage

3. **Remove low-quality data:**
   - Filter out empty lines
   - Filter very short sentences (<3 words)
   - Consider filtering very long sentences (>100 words)

### Medium Priority
4. **Optimize vocabulary size:**
   - Current vocab_size=32,000 but only 40-58% used
   - Consider reducing to 16,000-20,000 for better token density
   - Or increase training data to better utilize vocabulary

5. **Adjust max_len:**
   - Current max_len=32 truncates sentences at ~90th percentile
   - Consider increasing to 48-64 or filtering long sentences
   - Balance between memory usage and data loss

6. **Implement length-based filtering:**
   - Filter sentence pairs with extreme length ratios (>3:1 or <1:3)
   - Ensure reasonable alignment between source and target

### Implementation Plan

#### Phase 1: Data Cleaning Pipeline
1. Create enhanced preprocessing script with:
   - Character encoding normalization
   - Empty line removal
   - Length-based filtering
   - Special character handling

2. Update tokenizer training to use full dataset

#### Phase 2: Tokenizer Optimization
1. Experiment with reduced vocab_size (16k, 20k, 24k)
2. Compare tokenization quality and coverage
3. Select optimal vocabulary size

#### Phase 3: Dataset Refinement
1. Create filtered dataset with quality controls
2. Generate statistics for filtered dataset
3. Update training configuration accordingly

## Expected Impact
- **Improved translation quality:** Better character handling and cleaner data
- **Reduced vocabulary sparsity:** Higher token density improves learning
- **Better memory utilization:** Optimized max_len and batch_size
- **Faster convergence:** Cleaner data with fewer outliers

## Next Steps
1. Create enhanced preprocessing script
2. Run data cleaning pipeline
3. Retrain tokenizers with optimized parameters
4. Update training configuration
5. Test impact on validation loss and translation quality