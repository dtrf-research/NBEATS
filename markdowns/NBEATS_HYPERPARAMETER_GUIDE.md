# N-BEATs Hyperparameter Optimization Guide

## Architecture Overview
N-BEATs (Neural Basis Expansion Analysis with Time Series forecasting) is a pure deep learning architecture with:
- **Stacks**: Independent pathways that process the input differently
- **Layers**: Depth within each stack (3-5 typical)
- **Width**: Feature dimension at each layer
- **Input Context**: Historical lookback window

---

## Parameter Recommendations for Electricity Demand Forecasting

### 1. **Input Context (input_chunk_length)**

**Why it matters**: Determines how much historical data the model sees.

**Best practices**:
- Electricity demand has strong **daily**, **weekly**, and **seasonal** patterns
- General rule: 2x - 10x the forecast horizon
- For 96-step (1 day) forecast horizon: 192-960 steps acceptable
- **Key insight**: Include full daily + weekly cycle:
  - 1 week = 96 × 7 = 672 steps
  - 2 weeks = 96 × 14 = 1,344 steps

**Recommended range**: `[3, 5, 7, 10, 14]` days
- **3 days** (384 steps): Captures 72h patterns, good for short-term
- **5 days** (544 steps): Captures 5-day business cycle
- **7 days** (672 steps): Full weekly cycle (BEST for demand)
- **10 days** (960 steps): Extended patterns
- **14 days** (1,344 steps): 2-week cycle + weekend patterns

---

### 2. **Number of Layers (num_layers)**

**Why it matters**: Controls depth of representation in each stack.

**Best practices**:
- Too shallow (1-2): Underfitting
- **3-4 layers**: Standard, good balance
- 5 layers: More expressive, slower
- 6+: Diminishing returns, overfitting risk

**Recommended range**: `[3, 4, 5]`
- **3 layers**: Lightweight, fast training (baseline)
- **4 layers**: Standard (current default - good choice)
- **5 layers**: More capacity if data is abundant

---

### 3. **Layer Width (layer_widths)**

**Why it matters**: Feature dimension at each layer; controls model capacity.

**Best practices**:
- Should **scale with input_chunk_length**
- Too small: Limited capacity
- Too large: Overfitting, slower training
- Heuristic: `layer_width ≈ 0.5 × input_chunk_length` to `1 × input_chunk_length`

**Current issue**: `[32, 64]` is too small for demand forecasting!

**Recommended range**: `[128, 256, 512, 1024]`
- **128**: Smaller models, less overfitting (with 3 days context)
- **256**: Sweet spot (standard N-BEATs choice)
- **512**: Larger capacity (with 7+ days context)
- **1024**: For 14-day context with 5+ layers

**Pairing logic**:
| input_chunk_length | Recommended layer_widths |
|--------------------|---------------------------|
| 3 days (384)       | 128, 256                  |
| 5 days (544)       | 256, 512                  |
| 7 days (672)       | 256, 512, 1024            |
| 10 days (960)      | 512, 1024                 |
| 14 days (1344)     | 512, 1024                 |

---

### 4. **Number of Stacks (num_stacks)**

**Why it matters**: Parallel pathway depth; controls total representational capacity.

**Best practices**:
- **10-20**: Lightweight models
- **20-30**: Standard (good balance)
- **30-50**: Large models (require more data, GPU)
- **50+**: Rare; usually overfits

**Recommended range**: `[15, 20, 30, 40]`
- **15 stacks**: Lightweight (fast, for prototyping)
- **20 stacks**: Good for smaller datasets
- **30 stacks**: Standard (current default)
- **40 stacks**: Larger capacity (requires more data)

---

### 5. **Batch Size**

**Why it matters**: Training stability and convergence speed.

**Best practices**:
- Larger batch → smoother gradients but less frequent updates
- Smaller batch → noisier but faster convergence
- Dataset size matters: electricity demand has ~35,000 samples/year
- GPU memory limits typical batch sizes

**Recommended range**: `[64, 128, 256, 512]`
- **64**: For smaller GPUs or more frequent updates
- **128**: Good balance for most setups
- **256**: Standard (current default)
- **512**: For larger datasets or GPUs with enough VRAM

---

## Recommended HPO Grids

### **Option A: Quick Prototyping** (2-3 hours on GPU)
```python
HPO_GRID = {
    "input_chunk_length": [96*i + 18*4 for i in [3, 7]],  # 3, 7 days
    "num_layers": [4],
    "layer_widths": [256],
    "num_stacks": [20, 30],
    "batch_size": [256]
}
# Total combinations: 1 × 1 × 1 × 2 × 1 = 2 models
```

### **Option B: Balanced Search** (4-8 hours on GPU)
```python
HPO_GRID = {
    "input_chunk_length": [96*i + 18*4 for i in [3, 5, 7]],  # 3, 5, 7 days
    "num_layers": [4, 5],
    "layer_widths": [256, 512],
    "num_stacks": [20, 30],
    "batch_size": [256, 512]
}
# Total combinations: 3 × 2 × 2 × 2 × 2 = 48 models
```

### **Option C: Comprehensive Search** (12-24 hours on GPU)
```python
HPO_GRID = {
    "input_chunk_length": [96*i + 18*4 for i in [3, 5, 7, 10]],  # 3-10 days
    "num_layers": [3, 4, 5],
    "layer_widths": [256, 512, 1024],
    "num_stacks": [20, 30, 40],
    "batch_size": [128, 256, 512]
}
# Total combinations: 4 × 3 × 3 × 3 × 3 = 324 models (expensive!)
```

### **Option D: Focus on Context (RECOMMENDED)** (6-12 hours on GPU)
```python
HPO_GRID = {
    "input_chunk_length": [96*i + 18*4 for i in [3, 5, 7, 10, 14]],  # 3-14 days (weekly cycles)
    "num_layers": [3, 4, 5],
    "layer_widths": [256, 512],
    "num_stacks": [25, 30],
    "batch_size": [256]
}
# Total combinations: 5 × 3 × 2 × 2 × 1 = 60 models
# Rationale: Focus on INPUT CONTEXT (most critical for demand)
#           while keeping other params reasonable
```

---

## Individual Model Recommendations

### **Baseline** (Fast, Risk-free)
```python
input_chunk_length = 96 * 3 + 18 * 4  # 3 days
num_layers = 4
layer_widths = 256
num_stacks = 30
batch_size = 256
```

### **Production** (Balanced, Recommended)
```python
input_chunk_length = 96 * 7 + 18 * 4  # 7 days (full weekly cycle)
num_layers = 4
layer_widths = 512
num_stacks = 30
batch_size = 256
```

### **Large Model** (High Capacity, Requires Good Data)
```python
input_chunk_length = 96 * 10 + 18 * 4  # 10 days
num_layers = 5
layer_widths = 1024
num_stacks = 40
batch_size = 512
```

---

## Training Hyperparameters (Fixed)

These are good defaults for electricity demand:

```python
N_EPOCHS = 100                              # Usually stops early anyway
NR_EPOCHS_VAL_PERIOD = 1                   # Check validation every epoch
EARLY_STOPPING_PATIENCE = 10                # Stop if no improvement for 10 epochs
EARLY_STOPPING_MIN_DELTA = 1e-5            # Minimum improvement threshold
OUTPUT_CHUNK_LENGTH = 96                    # Forecast 1 day ahead
OUTPUT_SHIFT = 96                           # 6-hour shift (optional, use 0 for standard)
RANDOM_STATE = 47                          # Reproducibility
```

---

## Expected Performance

Based on N-BEATs literature and typical electricity demand datasets:

| Model Size | Context | MAPE Expected | Training Time |
|------------|---------|---------------|----------------|
| Small      | 3 days  | 5-8%          | 10-20 min     |
| Medium     | 7 days  | 3-5%          | 20-40 min     |
| Large      | 10-14 days | 2-4%        | 40-60 min     |

*MAPE = Mean Absolute Percentage Error (lower is better)*

---

## Tips for Success

1. **Always start with 7-day context**: Electricity is highly day-of-week dependent
2. **Layer width should match context**: Wider context → wider layers
3. **Increase complexity gradually**: Start with 4 layers, add 5th if needed
4. **Monitor validation loss**: Early stopping is your friend
5. **Save all models**: Each one provides useful insights
6. **Cross-validate**: Test on multiple non-overlapping periods
7. **Batch size matters less than context**: Focus on input_chunk_length first

---

## References

- **N-BEATs Paper**: https://arxiv.org/abs/1905.10437
- **Darts Library**: https://unit8co.github.io/darts/
- **Time Series Best Practices**: Include seasonal lags (weekly for demand)

