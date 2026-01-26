# How to Improve SimRegMatch Results

## Current Performance Analysis
- **Supervised Baseline**: MAE 5.83 (with 10% labeled data)
- **Best Semi-Supervised**: MAE 6.10 (experiment_16, 40% labeled)
- **Gap**: ~0.27-0.96 MAE worse than supervised baseline

## Key Issues Identified

### 1. **Hyperparameter Suboptimality**
- `lambda_u=0.01` is too low → unlabeled loss barely contributes
- `beta=0.6` may not be optimal for pseudo-label calibration
- `t=1.0` (temperature) may be too high for similarity matching
- `percentile=0.95` may be too permissive (accepts 95% of pseudo-labels)

### 2. **No Learning Rate Scheduling**
- Fixed LR throughout training → model may not converge optimally
- No warmup phase → training instability early on

### 3. **Pseudo-Label Quality Issues**
- Uncertainty estimation may not be robust enough
- Threshold updates every step → unstable filtering

### 4. **Loss Function Sensitivity**
- MSE loss is sensitive to outliers
- May benefit from L1 or Huber loss

### 5. **Augmentation Strength**
- Strong augmentation: `RandAugmentPC(n=2, m=10)` may be too weak
- Could try stronger augmentation for better regularization

---

## Recommended Improvements (Priority Order)

### **Priority 1: Hyperparameter Tuning** ⭐⭐⭐

**Immediate Actions:**
1. **Increase `lambda_u`**: Try `0.1`, `0.5`, `1.0` (currently 0.01)
   - The unlabeled loss is barely contributing to training
   - Higher values will give more weight to pseudo-labels

2. **Optimize `beta`**: Try `0.5`, `0.7`, `0.8` (currently 0.6)
   - Controls balance between model predictions and similarity-based pseudo-labels
   - Lower beta = more trust in similarity matching

3. **Lower temperature `t`**: Try `0.1`, `0.3`, `0.5` (currently 1.0)
   - Lower temperature = sharper similarity distribution
   - Helps focus on most similar labeled examples

4. **Tighten `percentile`**: Try `0.85`, `0.90` (currently 0.95)
   - More selective pseudo-label filtering
   - Only use most confident predictions

**Recommended Command:**
```bash
docker run -it --rm \
    --gpus '"device=2"' \
    -v $(pwd):/workspace \
    -v /work/ammar/sslrp/data:/workspace/data \
    -v /work/ammar/sslrp/SimRegMatch/results:/workspace/results \
    simregmatch:latest \
    python /workspace/main.py \
        --dataset utkface \
        --data_dir /workspace/data \
        --labeled-ratio 0.1 \
        --batch_size 32 \
        --epochs 200 \
        --lr 0.001 \
        --optimizer adam \
        --lambda-u 0.5 \
        --beta 0.7 \
        --t 0.3 \
        --percentile 0.90 \
        --gpu 0
```

---

### **Priority 2: Add Learning Rate Scheduling** ⭐⭐⭐

**Implementation Needed:**
- Add cosine annealing or step decay scheduler
- Consider warmup phase for first few epochs

**Benefits:**
- Better convergence
- More stable training
- Potentially better final performance

**Code Changes Required:**
- Add scheduler argument to `args.py`
- Initialize scheduler in `SimRegMatchTrainer.__init__()`
- Step scheduler in training loop

---

### **Priority 3: Improve Uncertainty Estimation** ⭐⭐

**Current Approach:**
- Uses variance across `iter_u=5` predictions
- Updates threshold every step

**Improvements:**
1. **Increase `iter_u`**: Try `10`, `15` (currently 5)
   - More Monte Carlo samples = better uncertainty estimate

2. **Smoother threshold updates**: 
   - Update threshold every N steps instead of every step
   - Use exponential moving average for threshold

3. **Alternative uncertainty metrics**:
   - Consider entropy-based uncertainty
   - Combine multiple uncertainty sources

---

### **Priority 4: Loss Function Alternatives** ⭐⭐

**Try L1 Loss:**
- Less sensitive to outliers than MSE
- May work better for age regression

**Try Huber Loss:**
- Combines benefits of L1 and MSE
- Robust to outliers while maintaining smooth gradients

**Command:**
```bash
--loss l1  # or --loss huber (if implemented)
```

---

### **Priority 5: Stronger Augmentation** ⭐

**Current:**
- `RandAugmentPC(n=2, m=10)` - 2 random augmentations with magnitude 10

**Try:**
- Increase `n` to 3-4 augmentations
- Increase `m` to 12-15 for stronger augmentation
- Modify `UTKFace_Unlabeled.get_strong()` method

---

### **Priority 6: Training Strategy Improvements** ⭐

1. **Warmup Phase**:
   - Start with lower learning rate for first 5-10 epochs
   - Gradually increase to full LR

2. **Curriculum Learning**:
   - Start with stricter pseudo-label filtering (higher percentile)
   - Gradually relax as training progresses

3. **Labeled Data Mixing**:
   - Ensure good balance between labeled and unlabeled batches
   - Consider increasing batch size for labeled data

---

## Experimental Plan

### Phase 1: Hyperparameter Grid Search
Test combinations of:
- `lambda_u`: [0.1, 0.5, 1.0]
- `beta`: [0.5, 0.7, 0.8]
- `t`: [0.1, 0.3, 0.5]
- `percentile`: [0.85, 0.90, 0.95]

**Best from experiments so far:** None beat supervised baseline

### Phase 2: Add Learning Rate Scheduling
- Implement cosine annealing
- Test with best hyperparameters from Phase 1

### Phase 3: Improve Uncertainty Estimation
- Increase `iter_u` to 10-15
- Implement smoother threshold updates

### Phase 4: Loss Function & Augmentation
- Test L1 loss
- Test stronger augmentation

---

## Quick Win: Try This First

**Most promising configuration based on analysis:**

```bash
docker run -it --rm \
    --gpus '"device=2"' \
    -v $(pwd):/workspace \
    -v /work/ammar/sslrp/data:/workspace/data \
    -v /work/ammar/sslrp/SimRegMatch/results:/workspace/results \
    simregmatch:latest \
    python /workspace/main.py \
        --dataset utkface \
        --data_dir /workspace/data \
        --labeled-ratio 0.1 \
        --batch_size 32 \
        --epochs 200 \
        --lr 0.001 \
        --optimizer adam \
        --lambda-u 0.5 \
        --beta 0.7 \
        --t 0.3 \
        --percentile 0.90 \
        --iter-u 10 \
        --loss l1 \
        --gpu 0
```

**Key Changes:**
- `lambda_u`: 0.01 → 0.5 (50x increase in unlabeled loss weight)
- `beta`: 0.6 → 0.7 (more trust in similarity matching)
- `t`: 1.0 → 0.3 (sharper similarity distribution)
- `percentile`: 0.95 → 0.90 (more selective pseudo-labels)
- `iter-u`: 5 → 10 (better uncertainty estimation)
- `loss`: mse → l1 (less sensitive to outliers)

---

## Expected Improvements

- **Hyperparameter tuning**: +0.2-0.5 MAE improvement
- **LR scheduling**: +0.1-0.3 MAE improvement
- **Better uncertainty**: +0.1-0.2 MAE improvement
- **L1 loss**: +0.1-0.2 MAE improvement

**Combined potential**: Could close gap to supervised baseline or even exceed it

---

## Monitoring Recommendations

Track these metrics during training:
1. **Pseudo-label quality**: Monitor threshold values and mask ratios
2. **Loss components**: Track labeled vs unlabeled loss separately
3. **Validation MAE**: Early stopping if not improving
4. **Uncertainty distribution**: Monitor how uncertainty evolves

---

## Notes

- The framework is fundamentally sound but needs better hyperparameters
- Semi-supervised learning is challenging - may need more epochs or different strategies
- Consider comparing against other SSL methods (FixMatch, MixMatch) for reference
