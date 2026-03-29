# Project Coral Sentinel - Mathematical Correlations & Physics Model

## Executive Summary

The `data_generator.py` script generates **500,000 rows** of synthetic oceanographic data with **highly non-linear relationships** between temperature, CO2, depth, and pH. These relationships are grounded in real oceanographic physics, forcing XGBoost to learn complex patterns rather than simple linear associations.

---

## Core Mathematical Models

### 1. TEMPERATURE MODEL: Thermocline Stratification + Seasonal Variation

#### 1.1 Depth-Dependent Temperature (Thermocline)

```
T(z) = T_abyssal + (T_surface - T_abyssal) × exp(-z / H_th)

Where:
  T_abyssal   = 4°C        (deep water baseline)
  T_surface   = 26°C       (tropical surface baseline)
  z           = depth (m)
  H_th        = 100m       (thermocline scale height)
```

**Why Non-Linear:**
- Exponential decay creates **curved relationship** (not linear) between depth and temperature
- A change of 10m at surface (0-10m) has ~7x more effect than at depth (500-510m)
- This teaches XGBoost that depth's importance is **context-dependent**

**Real-World Physics:**
Warm surface water creates a density barrier; below this thermocline, water temperature drops rapidly before stabilizing at abyssal depths.

#### 1.2 Seasonal Temperature Modulation

```
T_seasonal = 3.5 × sin(2π × DOY / 365) × exp(-z / 10)

Where:
  DOY = Day of year (0-365)
  Seasonal amplitude = 3.5°C at surface
```

**Why Non-Linear:**
- Seasonal effect is **depth-modulated**: full amplitude at surface, zero at depth >30m
- Introduces **temporal autocorrelation** (nearby days have similar temperatures)
- Seasonal amplitude multiplied by depth-dependent factor creates **multiplicative interaction**

#### 1.3 ENSO (El Niño Southern Oscillation) Simulation

```
ENSO(t) = 0.5 × sin(2π × t / 3.5) + 0.3 × sin(2π × t / 5.2)

Where:
  t = year
  3.5 year cycle = El Niño typical frequency
  5.2 year cycle = Quasi-Biennial Oscillation
```

**Why Non-Linear:**
- Two **incommensurate frequencies** create beats and amplitude modulation
- ENSO effect on temperature: `ΔT_ENSO = ENSO(t) × 2.0 × exp(-z/100)`
- Superposition of two cycles = quadratic complexity for pattern recognition

#### 1.4 Marine Heatwave Anomalies (Edge Case)

```
Probability(heatwave) = 5% of data points
ΔT_heatwave ~ Gamma(α=2, β=2)  [0-8°C spike]
```

**Why Challenging:**
- Rare events (5%) create **class imbalance**
- Heatwaves are **discrete** (either present or absent), not continuous
- Overlaps with seasonal/ENSO effects: XGBoost must disentangle multiple signals

#### 1.5 Upwelling Events (Cold Water Intrusion)

```
Probability(upwelling) = 3% of data
ΔT_upwelling ~ -Exponential(λ=0.5)  [-2 to 0°C cooling]
```

**Why Non-Linear:**
- Upwelling is **intermittent** (binary trigger with continuous magnitude)
- Competes with heatwave signal at surface: model learns **conditional logic**
- Increases CO2 simultaneously (below)

---

### 2. CO2 MODEL: Remineralization, Solubility, & Photosynthesis

#### 2.1 Baseline CO2 (Keeling Curve)

```
CO2_baseline = 380 + 1.5 × year

Where:
  380 ppm = 2018 baseline
  1.5 ppm/year = anthropogenic increase rate
```

**Why Non-Linear:**
- Linear trend, but **interacts multiplicatively** with other effects below
- Ensures realistic **temporal drift** over 5-year dataset

#### 2.2 Depth-Dependent Remineralization

```
CO2_remineralization = 15 × tanh(z / 200)

Where:
  z = depth (m)
  tanh() = saturating hyperbolic tangent function
```

**Why Highly Non-Linear:**
- **Saturation function**: CO2 increases rapidly 0-300m, then plateaus
- Deep water (>1000m): asymptotes to +15 ppm above baseline
- Surface water (0-100m): nearly 0 contribution
- Creates **discontinuous gradient**: same depth difference has different CO2 change depending on absolute depth

**Real-World Physics:**
Remineralization is the oxidation of sinking organic matter. High rates near productive surface, exhausted at depth.

#### 2.3 Temperature Effect on CO2 Solubility (Henry's Law)

```
CO2_solubility = 0.04 × (T - 25°C)^1.5

Where:
  T = temperature (°C)
  Exponent = 1.5 (not 1!)
```

**Why Non-Linear:**
- **Power-law relationship** (exponent 1.5, not linear)
- Cold water (T=4°C): CO2_solubility ≈ 0.04 × (-21)^1.5 ≈ -18 ppm
- Warm water (T=30°C): CO2_solubility ≈ 0.04 × 5^1.5 ≈ +0.45 ppm
- **Asymmetry**: cooling effect is much stronger than warming effect
- XGBoost must learn **direction-dependent sensitivity**

**Real-World Physics:**
Warm water holds less dissolved CO2 (inverse solubility). The 1.5 power accounts for non-ideal gas behavior.

#### 2.4 Seasonal CO2 Drawdown (Photosynthesis)

```
CO2_photosynthesis = -8 × sin(2π × DOY / 365) × exp(-z / 50)

Where:
  DOY = day of year
  Photic zone depth ≈ 50m
  Amplitude = 8 ppm
```

**Why Non-Linear:**
- **Sinusoidal modulation**: effect oscillates with annual period
- **Exponential depth decay**: photosynthesis only in photic zone
- Interaction: `∂CO2/∂DOY` changes with depth
- Peak drawdown at surface in summer, minimal at depth

#### 2.5 Upwelling CO2 Boost

```
Probability(upwelling) = 3%
ΔCO2_upwelling ~ Uniform(20, 60 ppm)
```

**Why Correlative:**
- **Coupled with temperature**: same event triggers both cooling AND CO2 increase
- XGBoost must recognize that temperature ↓ + CO2 ↑ = upwelling signal
- Creates **negative correlation** between T and CO2 in upwelling regions (unusual!)

---

### 3. pH MODEL: Carbonate Chemistry & Alkalinity

#### 3.1 Carbonate System Equilibria

The full model solves:
```
[H+] + [Na+] = [HCO3-] + 2[CO3--] + [OH-]

Dissociation constants (temperature-dependent):
  K1 = 10^(-6.35 + 0.0107 × (T - 20))
  K2 = 10^(-10.33 + 0.0018 × (T - 20))
  Kw = 10^(-14.0 - 0.018 × (T - 20))
```

**Why Highly Non-Linear:**
- **Logarithmic scale**: pH = -log10[H+]
- **Equilibrium constants change with temperature** (exponential)
- System of **coupled non-linear equations** (iteratively solved)
- Small changes in CO2 cause **non-proportional pH changes** (buffering effect)

#### 3.2 DIC-to-Alkalinity Relationship

```
Alk = [HCO3-] + 2[CO3--] + [OH-] - [H+]

Where total alkalinity is:
  Alk ≈ 2320 - 5.4 × (T - 20) μmol/kg
```

**Why Non-Linear:**
- Alkalinity **decreases with temperature** (coefficient -5.4)
- Interaction with pH: same CO2 at different T → different pH
- Buffering capacity: Alk determines how much CO2 change it takes to shift pH

#### 3.3 Depth-Dependent Acidification

```
ΔpH_depth = -0.0002 × depth

Where:
  Each 100m deeper → 0.02 unit lower pH
  3000m deep → 0.6 unit lower pH
```

**Why Important:**
- **Cumulative with other factors**: adds non-linearity
- Deep water already close to saturation horizon (CaCO3 solubility limit)
- Deep + warm + high CO2 = extreme acidification risk

#### 3.4 Temperature Effect on pH Buffering

```
ΔpH_temp = -0.01 × (T - 25°C)

Where:
  Warm water (30°C): pH ↓ 0.05
  Cold water (4°C): pH ↑ 0.21
```

**Why Non-Linear:**
- Buffering capacity changes with temperature
- Warm coral reefs (T≈27°C) are especially vulnerable
- Interaction: Temperature affects both CO2 solubility AND buffering

---

## Cross-Feature Correlations & Interactions

### Feature Interaction Matrix

| Interaction | Type | Strength | Formula |
|---|---|---|---|
| **Depth × Temperature** | Multiplicative | Very High | `exp(-z/H) × (T-T_base)` |
| **Temp × CO2** | Non-linear | High | `0.04 × (T-25)^1.5 + remineralization(z)` |
| **Temp × pH** | Logarithmic | Very High | Through carbonate equilibrium |
| **CO2 × pH** | Logarithmic | Very High | pH = -log10([H+]) depends on CO2 |
| **Depth × pH** | Saturating | High | `-0.0002 × depth` |
| **Seasonal × Depth** | Multiplicative | Medium | `seasonal × exp(-z/10)` |
| **ENSO × Temp** | Multiplicative | Medium | `ENSO × 2.0 × exp(-z/100)` |

### Why XGBoost Will Struggle (and Learn Deep Patterns):

1. **Multiplicative Interactions**: Most correlations use `×` (multiplication), not `+` (addition)
   - Linear models fail here; tree-based models excel
   - Decision trees can approximate `exp()` and `tanh()` with many splits

2. **Depth Modulation**: Almost every feature's variance changes with depth
   - Same seasonal amplitude = 3.5°C at surface, 0°C at 100m
   - Interaction terms: `feature × exp(-depth/scale)`
   - Requires learning **depth-dependent feature importance**

3. **Saturation Functions**: `tanh()` and exponential decay
   - Remineralization: `tanh(z/200)` → derivative decreases with depth
   - Heatwaves: `exp(-z/10)` → exponential decay with depth
   - Tree-based model approximates via step functions

4. **Temporal Autocorrelation**: Time-series structure
   - Seasonal patterns (365-day cycle)
   - ENSO patterns (3.5 and 5.2-year cycles)
   - Heatwaves clustered in time
   - Creates **non-i.i.d.** data that penalizes overfitting

5. **Rare Event Clustering**: Edge cases (heatwaves, upwelling)
   - Heatwaves: 5% overall, but clustered temporally
   - Creates **conditional probability shifts**
   - XGBoost must learn: "IF season=summer AND ENSO=El Niño, THEN heatwave prob ↑"

---

## Dataset Composition

### Size: 500,000 rows

```
Training set: ~400,000 rows (80%)
Validation:   ~50,000 rows (10%)
Test:         ~50,000 rows (10%)
```

### Feature Engineering (Included)

The script automatically creates derived features:

```python
temperature_squared     = T^2          # Capture non-linearity
co2_sqrt               = √(CO2)        # Dimensionality reduction
depth_log              = log(1 + depth) # Handle skewness
temp_depth_interaction = T × log(depth) # Explicit interaction
co2_temp_ratio         = CO2 / (T+ε)   # Normalized ratio
```

**Why Pre-engineer:**
- Helps XGBoost (trees split on original features, but engineered features accelerate convergence)
- Explainability: model can directly show that `temp_depth_interaction` is important

---

## Anomaly Distribution

| Anomaly Type | Probability | Magnitude | Effect on Target |
|---|---|---|---|
| Heatwave | 5% | +2 to +8°C | pH ↓ 0.1-0.3 |
| Upwelling | 3% | -2 to 0°C, +20-60 ppm CO2 | pH ↓ 0.2-0.5 |
| Acidification hotspot | 2% | CO2 ↑30-80 ppm | pH ↓ 0.2-0.5 |
| Hypoxia | 1% | T ↓1-3°C, pH ↓0.15-0.35 | Stress indicator |
| Hydrothermal vent | 0.1% | T → extreme, pH ↓ | Ultra-rare |

---

## Why This Dataset is Ideal for XGBoost

### ✅ Advantages:

1. **Non-Linearity**: Every feature pair has non-linear relationships
2. **Interactions**: Multiplicative, exponential, logarithmic (tree splits naturally approximate)
3. **Temporal Structure**: Seasonal patterns force model to learn periodicities
4. **Rare Events**: 5-0.1% anomalies create imbalanced classification scenarios
5. **Realistic Physics**: Correlations match real oceanography → model learns transferable patterns

### ⚠️ Challenges:

1. **Autocorrelation**: Time-series structure requires careful train/test split
2. **Multicollinearity**: Many derived features are correlated by design
3. **Curse of Dimensionality**: 16+ features after engineering
4. **Imbalanced Targets**: If predicting "bleaching risk" from anomalies, heavy class imbalance

---

## Quick Start Guide

```bash
# Install dependencies
pip install numpy pandas scipy

# Generate data
python data_generator.py

# Output: massive_coral_training_data.csv (500,000 rows × 16 columns)
# File size: ~150 MB

# Use with XGBoost
import xgboost as xgb
df = pd.read_csv('massive_coral_training_data.csv')
X = df[['temperature', 'co2', 'depth', 'temperature_squared', 'co2_sqrt', ...]]
y = df['ph']  # or engineer your own target (bleaching risk = f(pH, T))

model = xgb.XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.01)
model.fit(X, y)
```

---

## References & Real-World Grounding

### Physics Models:
- **Thermocline**: Levitus climatology (NOAA)
- **Carbonate Chemistry**: Zeebe & Wolf-Gladrow (2001)
- **CO2 Solubility**: Weiss (1974)
- **ENSO Cycles**: CPC Oceanic Niño Index (ONI)

### Data Realism:
- Temperature range: -2 to 35°C (global oceans)
- CO2 range: 200-800 ppm (pre-industrial to extreme acidification)
- Depth range: 0.5 to 3000m (surface to abyssal)
- pH range: 7.2 to 8.4 (ocean typical)

---

## Summary

Your XGBoost model will learn:
1. **Depth-dependent feature importance** (features matter more at surface)
2. **Seasonal patterns** (365-day periodicity in all variables)
3. **Rare event detection** (heatwaves, upwelling)
4. **Non-linear feature interactions** (exp, tanh, power laws)
5. **Carbonate chemistry** (implicit through pH-CO2-T correlations)

This is **enterprise-grade training data** for a production coral reef monitoring system! 🪸

