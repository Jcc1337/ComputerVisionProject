# Trigger Metrics Analysis Guide for Backdoor Attack Paper

## Overview

The `calculate_shape_entropy.py` script analyzes 18 trigger variants (3 shapes × 6 color types) and produces:

1. **Console output** with detailed statistics grouped by shape
2. **CSV file** (`trigger_metrics.csv`) for correlation analysis with model performance

## Key Metrics for Your Paper

### 1. Shannon Entropy (Value Distribution)

- **What it measures**: How varied the RGB colors are in the trigger
- **For your paper**: Use in scatter plot: **Shannon Entropy vs. ASR**
- **Interpretation**:
  - High entropy (e.g., 3.5+): Diverse colors (inverted, mean-shifted)
  - Low entropy (<2.0): Uniform colors (mean, blurred)
- **Column in CSV**: `entropy_rgb`

### 2. Spatial Entropy (Edge/Transition Distribution)

- **What it measures**: How complex the spatial structure is (via Laplacian filtering)
- **For your paper**: Use in scatter plot: **Spatial Entropy vs. ASR**
- **Interpretation**:
  - High spatial entropy: Random shapes, irregular patterns
  - Low spatial entropy: Smooth, uniform patterns
- **Column in CSV**: `spatial_entropy`

### 3. Edge Density (% of pixels that are transitions)

- **What it measures**: Percentage of pixels that are edges/transitions
- **For your paper**: Use in scatter plot: **Edge Density vs. Stealthiness (LPIPS)**
- **Interpretation**:
  - High edge density (>20%): Triggers have clear boundaries
  - Low edge density (<5%): Smooth, blended triggers
- **Column in CSV**: `edge_density`

### 4. Perimeter-to-Area Ratio (P/A) - **Critical for NNI**

- **What it measures**: Perimeter / Area - contact surface for contextual aggregation
- **For your paper**: Use in scatter plot: **P/A Ratio vs. ASR**
- **Why important**: This is THE core principle of your NNI-based backdoor attack
  - Higher P/A = more "contact" with neighboring pixels
  - May correlate directly with attack success
- **Column in CSV**: `perimeter_area_ratio`

### 5. Contrast Ratio

- **What it measures**: (Max - Min) / (Max + Min) - brightness variation
- **For your paper**: Design family clustering (See Table recommendations below)
- **Column in CSV**: `contrast_contrast_ratio`

### 6. Frequency Content (FFT Analysis)

- **What it measures**: Distribution of spatial frequencies
  - Low Freq: Smooth, geometric shapes
  - High Freq: Fine details, noise-like triggers
- **For your paper**: Differentiates "smooth" vs "noisy" designs
- **Columns in CSV**: `freq_low_freq_percent`, `freq_mid_freq_percent`, `freq_high_freq_percent`

---

## CSV Format for Analysis

The script generates `trigger_metrics.csv` with these key columns:

```
filename, shape, color_variant, entropy_rgb, spatial_entropy, edge_density,
perimeter_area_ratio, contrast_contrast_ratio, freq_high_freq_percent, ...
```

### How to Use in Python/R for Correlation:

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load metrics
df = pd.read_csv('trigger_metrics.csv')

# Combine with your training results
# (You'll need: ASR, LPIPS/Stealthiness, mIoU for each trigger)
results_df = pd.merge(df, your_training_results, on='filename')

# Example: Correlate entropy with ASR
corr, p_value = stats.pearsonr(results_df['entropy_rgb'], results_df['ASR'])
print(f"Shannon Entropy vs. ASR: r={corr:.3f}, p={p_value:.4f}")

# Plot
plt.scatter(results_df['entropy_rgb'], results_df['ASR'])
plt.xlabel('Shannon Entropy')
plt.ylabel('Attack Success Rate (%)')
plt.title('Higher Entropy → Higher ASR?')
```

---

## Table Templates for Your Paper

### Table A: Trigger Design Families & Performance Summary

```
Design Family          Avg ASR    Avg mIoU    Avg LPIPS    Avg Edge%    Avg Entropy
─────────────────────────────────────────────────────────────────────────────────
Random Pixels          XX%        XX.X%       X.XXX        XX%          X.XXX
Blurred Colors         XX%        XX.X%       X.XXX        XX%          X.XXX
Inverted Colors        XX%        XX.X%       X.XXX        XX%          X.XXX
Mean Color             XX%        XX.X%       X.XXX        XX%          X.XXX
Mean-Shifted           XX%        XX.X%       X.XXX        XX%          X.XXX
See-Through (Roads)    XX%        XX.X%       X.XXX        XX%          X.XXX
```

### Table B: Shape Comparison (Circle vs. Cross vs. Random)

```
Shape      Avg P/A    Avg Edge%    High ASR?    Notes
────────────────────────────────────────────────────────────
Circle     X.XXX      XX%          [YES/NO]     Smooth geometry
Cross      X.XXX      XX%          [YES/NO]     Moderate complexity
Random     X.XXX      XX%          [YES/NO]     Highest edge density
```

### Table C: Top vs. Bottom Performers

Focus on **5 best and 5 worst triggers** by ASR:

```
Rank    Trigger Type        Shape    Shannon    P/A      ASR      LPIPS    Notes
────────────────────────────────────────────────────────────────────────────────
1       Mean-Shifted        Random   X.XXX      X.XXX    X.XX%    X.XXX    Best attack
2       Inverted            Circle   X.XXX      X.XXX    X.XX%    X.XXX
...
-1      Blurred             Cross    X.XXX      X.XXX    X.XX%    X.XXX    Worst attack
-2      See-Through         Circle   X.XXX      X.XXX    X.XX%    X.XXX
```

---

## Figures/Plots Recommended

### Figure 1: Spatial Entropy vs. ASR (Scatter + Regression)

```python
x = results_df['spatial_entropy']
y = results_df['ASR']
colors = results_df['color_variant'].map({'BLURRED': 'red', ...})

plt.scatter(x, y, c=colors, s=100, alpha=0.6)
# Add regression line
z = np.polyfit(x, y, 1)
plt.plot(x, np.poly1d(z)(x), "k--", linewidth=2)
plt.xlabel('Spatial Entropy')
plt.ylabel('Attack Success Rate (%)')
```

### Figure 2: Edge Density vs. Stealthiness (LPIPS)

- Shows the **stealthiness-effectiveness trade-off**
- Identify "sweet spot" triggers that are both stealthy AND effective

### Figure 3: Perimeter-to-Area Ratio vs. ASR (Colored by Shape)

- Validates that **higher P/A correlates with better attacks** (if true)
- Evidence that NNI context aggregation benefits from larger trigger perimeters

### Figure 4: Qualitative Success vs. Failure

- Side-by-side: Original Image | Poisoned Image | Model Output
- Show one high-ASR trigger and one low-ASR trigger
- Highlights how spatial entropy affects visual "appearance"

---

## Column Reference (Full CSV Schema)

| Column                  | Meaning                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| filename                | Trigger image filename                                           |
| shape                   | circle, cross, or random                                         |
| color_variant           | RANDOM-PIXELS, BLURRED, INVERTED, MEAN, MEAN-SHIFTED, SEETHROUGH |
| entropy_rgb             | Shannon entropy of RGB values (VALUE distribution)               |
| entropy_red/green/blue  | Per-channel Shannon entropy                                      |
| spatial_entropy         | Spatial entropy (STRUCTURE complexity via Laplacian)             |
| spatial_entropy_masked  | Spatial entropy of masked region only                            |
| edge_density            | % of pixels that are edges (gradient > threshold)                |
| mean_gradient           | Average gradient magnitude                                       |
| contrast_mean_val       | Mean pixel value (brightness)                                    |
| contrast_std_dev        | Standard deviation (variation)                                   |
| contrast_dynamic_range  | Max - Min pixel values                                           |
| contrast_contrast_ratio | (Max-Min)/(Max+Min) normalized contrast                          |
| freq_low_freq_percent   | % of frequency energy in low frequencies                         |
| freq_mid_freq_percent   | % of frequency energy in mid frequencies                         |
| freq_high_freq_percent  | % of frequency energy in high frequencies                        |
| shape_coverage_percent  | % of image covered by trigger                                    |
| shape_circularity       | 1.0=perfect circle, <1 = less circular                           |
| perimeter_area_ratio    | **P/A ratio** - key for NNI contact analysis                     |
| active_pixels           | Number of non-transparent pixels                                 |
| total_pixels            | Total pixels (55×55=3025)                                        |

---

## Next Steps

1. **Run training** with these 18 trigger variants
2. **Collect ASR, LPIPS, mIoU** for each trigger
3. **Merge** training results with `trigger_metrics.csv`
4. **Create correlation plots** using template code above
5. **Identify patterns**:
   - Which metric best predicts ASR?
   - Is there a stealthiness-effectiveness trade-off?
   - Do certain shapes outperform others?
6. **Write paper section**:
   - "Trigger Design Analysis"
   - "Why Spatial Entropy Matters for Backdoor Attacks"
   - "The Role of Perimeter-to-Area in Context Aggregation"

---

## Questions This Analysis Answers

✓ Which trigger design is most effective? (highest ASR)  
✓ Which is most stealthy? (lowest LPIPS)  
✓ Is there a sweet spot? (high ASR + low LPIPS)  
✓ Do geometric shapes matter? (circle vs. cross vs. random)  
✓ Does color type matter? (blurred vs. inverted vs. mean-shifted)  
✓ Why do certain triggers work better? (correlate metrics with success)  
✓ Can we predict performance from trigger properties? (regression analysis)
