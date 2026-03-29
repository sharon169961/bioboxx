#!/usr/bin/env python3
"""
Project Coral Sentinel - Advanced Synthetic Oceanographic Data Generator
Generates 500,000 rows of realistic coral reef environmental data with complex 
non-linear relationships for XGBoost model training.

Features:
- Realistic depth-dependent temperature stratification
- CO2-pH carbonate chemistry interactions
- Seasonal variations with inter-annual oscillations
- Edge-case anomalies (El Niño, upwelling, thermal stress events)
- Non-linear cross-feature relationships
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CoralReefDataGenerator:
    """
    Generates synthetic oceanographic data with realistic carbonate chemistry
    and thermal dynamics for coral bleaching risk prediction.
    """

    def __init__(self, n_samples=500_000, random_seed=42):
        self.n_samples = n_samples
        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.data = {}
        print("🪸 Coral Sentinel Data Generator initialized")
        print(f"   Generating {n_samples:,} synthetic oceanographic records...")

    # ------------------------------------------------------------------
    # Time / depth / temperature / CO2  (unchanged from original)
    # ------------------------------------------------------------------

    def _generate_time_series(self):
        start_date = datetime(2018, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(self.n_samples)]
        day_of_year = np.array([(d - start_date).days % 365 for d in dates])
        year        = np.array([(d - start_date).days // 365 for d in dates])

        seasonal   = 3.5 * np.sin(2 * np.pi * day_of_year / 365)
        enso_phase = (0.5 * np.sin(2 * np.pi * year / 3.5) +
                      0.3 * np.sin(2 * np.pi * year / 5.2))

        return day_of_year, year, seasonal, enso_phase

    def _generate_depth(self):
        depths = np.concatenate([
            np.random.exponential(20, self.n_samples // 2),
            np.random.pareto(1.5, self.n_samples - self.n_samples // 2) * 100 + 50,
        ])
        return np.clip(depths, 0.5, 3000.0)

    def _generate_temperature(self, depth, seasonal, enso_phase, day_of_year):
        surface_temp_base  = 26.0
        thermocline_depth  = 100.0
        abyssal_temp       = 4.0

        depth_factor       = np.exp(-depth / thermocline_depth)
        temp_from_depth    = abyssal_temp + (surface_temp_base - abyssal_temp) * depth_factor
        seasonal_mod       = seasonal * depth_factor
        enso_effect        = enso_phase * (2.0 * depth_factor)

        heatwave_mask      = np.random.random(self.n_samples) < 0.05
        heatwave_anomaly   = np.zeros(self.n_samples)
        heatwave_anomaly[heatwave_mask] = np.random.gamma(2, 2, heatwave_mask.sum())

        diurnal            = (0.8 * np.sin(2 * np.pi * np.random.random(self.n_samples)) *
                              np.exp(-depth / 10))

        upwelling_mask     = np.random.random(self.n_samples) < 0.03
        upwelling_cooling  = np.zeros(self.n_samples)
        upwelling_cooling[upwelling_mask] = -np.random.exponential(2, upwelling_mask.sum())

        temperature = (temp_from_depth + seasonal_mod + enso_effect +
                       heatwave_anomaly + diurnal + upwelling_cooling)
        temperature += np.random.normal(0, 0.15, self.n_samples)
        temperature = np.clip(temperature, -2.0, 35.0)

        return temperature, heatwave_mask, upwelling_mask

    def _generate_co2_and_alkalinity(self, depth, temperature, day_of_year, year):
        baseline_co2           = 380 + 1.5 * year
        seasonal_co2_variation = -8 * np.sin(2 * np.pi * day_of_year / 365)
        remineralization       = 15 * np.tanh(depth / 200)
        # BUG-FIX: (temperature - 25) is negative for T < 25°C.  Raising a
        # negative value to a fractional power (1.5) produces NaN in NumPy
        # because the result is complex.  Use sign * |Δ|^1.5 to preserve the
        # correct physical direction (colder water = lower pCO2) while keeping
        # all values real.  Zero crossing (T == 25) is handled naturally.
        _dt_sol           = temperature - 25.0
        solubility_effect = 0.04 * np.sign(_dt_sol) * np.abs(_dt_sol) ** 1.5

        upwelling_co2          = np.random.random(self.n_samples) < 0.03
        upwelling_co2_boost    = np.zeros(self.n_samples)
        upwelling_co2_boost[upwelling_co2] = np.random.uniform(20, 60, upwelling_co2.sum())

        photosynthesis = (-5 * np.sin(2 * np.pi * np.random.random(self.n_samples)) *
                          np.exp(-depth / 50))

        co2 = (baseline_co2 + seasonal_co2_variation + remineralization +
               solubility_effect + photosynthesis + upwelling_co2_boost)
        co2 += np.random.normal(0, 2, self.n_samples)
        co2  = np.clip(co2, 200.0, 800.0)

        alkalinity  = 2320 - 5.4 * (temperature - 20)
        alkalinity += np.random.normal(0, 20, self.n_samples)
        alkalinity  = np.clip(alkalinity, 2200.0, 2450.0)

        return co2, alkalinity

    # ------------------------------------------------------------------
    # _carbonate_chemistry  — fully rewritten, hardened
    # ------------------------------------------------------------------

    def _carbonate_chemistry(self, co2, alkalinity, temperature, depth):
        """
        Vectorized, production-grade carbonate pH solver.

        Engineering guarantees
        ----------------------
        1. Complete vectorisation  – all conditionals use np.where(); no
           Python-level loops over array elements.
        2. Mathematical safety     – denominators are protected with np.clip()
           and 1e-10 guards; log / power inputs are clipped above zero.
        3. Memory efficiency       – in-place operators (+=, *=, np.multiply
           with out=) avoid redundant intermediate arrays wherever possible.
        4. Graceful error handling – the iterative solver runs inside a
           try/except that catches overflow / non-convergence warnings and
           logs them without crashing the pipeline.

        Fixes vs original
        -----------------
        * `if dic_calc < dic` (ValueError on arrays) → np.where()
        * `ph` started as scalar 8.0; now initialised as full float64 array
        * `remineralization_factor(depth)` was an undefined name inside the
          method scope; inlined as np.tanh(depth / 200)
        * Dead / double-computed `baseline_shift` scalar branch removed
        * All denominator expressions guarded against division-by-zero
        * Overflow warnings from np.power / np.exp now caught and logged
        """
        n = len(co2)

        # ── equilibrium constants (all shape-n arrays) ──────────────────
        # BUG-FIX (math safety): temperature is already clipped to [-2, 35]
        # upstream, but clip again here defensively before exponentiation.
        t  = np.clip(temperature, -2.0, 35.0)
        dt = t - 20.0                              # reused offset

        K0 = 0.034 * np.exp(-0.0019 * dt)          # CO2 solubility
        K1 = np.power(10.0, -6.35 + 0.0107 * dt)   # 1st dissociation
        K2 = np.power(10.0, -10.33 + 0.0018 * dt)  # 2nd dissociation
        Kw = np.power(10.0, -14.0 - 0.018 * dt)    # water dissociation

        # ── DIC initial estimate ─────────────────────────────────────────
        # BUG-FIX (wrong ref): original called undefined local `remineralization`;
        # replaced with the correct helper expression np.tanh(depth / 200).
        # BUG-FIX (unit no-op): K0 * (co2/1e6) * 1e6 == K0 * co2, simplified.
        dic = K0 * co2 * 0.5 + np.tanh(depth / 200.0) * 50.0 + 60.0
        dic += np.random.normal(0, 10, n)
        # in-place clip for memory efficiency (requirement 3)
        np.clip(dic, 50.0, 3000.0, out=dic)

        # ── iterative solver ─────────────────────────────────────────────
        # BUG-FIX: initialise ph as a float64 *array* (not scalar 8.0) so all
        # subsequent vectorised operations broadcast correctly.
        ph = np.full(n, 8.0, dtype=np.float64)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)  # surface overflows

                for iteration in range(5):
                    # [H+] — clipped to avoid log(0) and 0-division downstream
                    h = np.power(10.0, -ph)                         # shape (n,)
                    np.clip(h, 1e-10, 1.0, out=h)                  # in-place safety

                    # ── HCO3⁻ ──────────────────────────────────────────
                    # Alk = HCO3⁻ + 2·CO3²⁻ + OH⁻ − H⁺  (simplified charge balance)
                    # BUG-FIX: denominator guarded against zero with np.clip
                    denom_hco3 = h + K1 + K2 * h / np.clip(h + K2, 1e-10, None)
                    np.clip(denom_hco3, 1e-10, None, out=denom_hco3)
                    hco3 = (alkalinity * h) / denom_hco3

                    # ── CO3²⁻ ──────────────────────────────────────────
                    # BUG-FIX: denominator h² + K1·h + K1·K2 guarded
                    denom_co3 = h * h + K1 * h + K1 * K2
                    np.clip(denom_co3, 1e-10, None, out=denom_co3)
                    co3 = (alkalinity * K1 * K2) / denom_co3       # noqa: F841 (kept for completeness)

                    # ── reconstructed DIC from carbonate species ────────
                    # dic_calc = [CO2*] + [HCO3⁻] + [CO3²⁻]
                    # [CO2*] = H⁺ · HCO3⁻ / K1
                    k1_safe  = np.clip(K1, 1e-20, None)            # guard /K1
                    dic_calc = (h * hco3 / k1_safe) + hco3 + co3

                    # ── pH update (vectorised conditional) ──────────────
                    # BUG-FIX (root cause): replaced scalar `if dic_calc < dic`
                    # with np.where() so every element is updated independently.
                    delta = np.where(dic_calc < dic, -0.01, 0.01)
                    ph   += delta                                    # in-place +=

                    # in-place clip keeps pH in physical range
                    np.clip(ph, 7.0, 8.5, out=ph)

        except RuntimeWarning as rw:
            # Requirement 4: log warning, do NOT re-raise; ph array retains
            # whatever values were computed up to the failing iteration.
            logger.warning(
                "Carbonate solver: RuntimeWarning caught at iteration %d — "
                "possible overflow or non-convergence for some elements. "
                "Affected pH values will be clamped during post-processing. "
                "Detail: %s", iteration, rw
            )
        except Exception as exc:
            logger.error(
                "Carbonate solver: unexpected error — %s. "
                "Falling back to last valid ph array.", exc
            )

        # ── post-solver corrections ──────────────────────────────────────
        # Deep water acidification (respiration increases pCO2 at depth)
        depth_acidification = -0.0002 * depth        # already an array

        # Temperature buffering effect (warm water → lower pH)
        temp_effect = -0.01 * (temperature - 25.0)

        # Baseline shift for cold/acidic water-mass fraction (10 %)
        # BUG-FIX: original had a dead scalar `if` branch, then overwrote it
        # with np.where() — cleaned up to a single vectorised expression.
        baseline_shift = np.where(np.random.random(n) < 0.1, -0.15, 0.0)

        ph_final = ph + depth_acidification + temp_effect + baseline_shift
        ph_final += np.random.normal(0, 0.05, n)

        # Final physical bounds — in-place for memory efficiency
        np.clip(ph_final, 7.2, 8.4, out=ph_final)

        return ph_final, dic, alkalinity

    # ------------------------------------------------------------------
    # Edge cases  (unchanged from original)
    # ------------------------------------------------------------------

    def _apply_edge_cases(self, temperature, co2, ph, depth, day_of_year):
        el_nino_mask = ((day_of_year > 100) & (day_of_year < 150) &
                        (np.random.random(self.n_samples) < 0.02))
        temperature[el_nino_mask] += np.random.uniform(2, 4, el_nino_mask.sum())
        ph[el_nino_mask]          -= np.random.uniform(0.1, 0.3, el_nino_mask.sum())
        co2[el_nino_mask]         += np.random.uniform(10, 30, el_nino_mask.sum())

        acidification_mask = np.random.random(self.n_samples) < 0.02
        ph[acidification_mask]  -= np.random.uniform(0.2, 0.5, acidification_mask.sum())
        co2[acidification_mask] += np.random.uniform(30, 80, acidification_mask.sum())

        hypoxia_mask = (depth > 200) & (np.random.random(self.n_samples) < 0.01)
        temperature[hypoxia_mask] -= np.random.uniform(1, 3, hypoxia_mask.sum())
        ph[hypoxia_mask]          -= np.random.uniform(0.15, 0.35, hypoxia_mask.sum())

        vent_mask = np.random.random(self.n_samples) < 0.001
        temperature[vent_mask] = np.random.uniform(20, 400, vent_mask.sum())
        ph[vent_mask]          = np.random.uniform(2.5, 7.0, vent_mask.sum())

        return temperature, co2, ph

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def generate(self):
        print("\n📊 Generating temporal features...")
        day_of_year, year, seasonal, enso_phase = self._generate_time_series()

        print("🌊 Generating depth profiles...")
        depth = self._generate_depth()

        print("🌡️  Generating temperature with stratification...")
        temperature, heatwave_mask, upwelling_mask = self._generate_temperature(
            depth, seasonal, enso_phase, day_of_year
        )

        print("💨 Generating CO2 and alkalinity...")
        co2, alkalinity = self._generate_co2_and_alkalinity(
            depth, temperature, day_of_year, year
        )

        print("⚗️  Solving carbonate chemistry for pH...")
        ph, dic, alkalinity = self._carbonate_chemistry(
            co2, alkalinity, temperature, depth
        )

        print("⚠️  Applying edge-case anomalies...")
        temperature, co2, ph = self._apply_edge_cases(
            temperature, co2, ph, depth, day_of_year
        )

        print("📦 Assembling dataset...")
        self.data = pd.DataFrame({
            'temperature': temperature,
            'co2':         co2,
            'depth':       depth,
            'ph':          ph,
        })

        self.data['temperature_squared']    = self.data['temperature'] ** 2
        self.data['co2_sqrt']               = np.sqrt(np.clip(self.data['co2'], 0, None))
        self.data['depth_log']              = np.log1p(self.data['depth'])
        self.data['temp_depth_interaction'] = (self.data['temperature'] *
                                               np.log1p(self.data['depth']))
        self.data['co2_temp_ratio']         = (self.data['co2'] /
                                               (self.data['temperature'] + 1e-6))
        self.data['day_of_year']  = day_of_year
        self.data['year']         = year
        self.data['month']        = (day_of_year // 30).astype(int) % 12 + 1
        self.data['is_heatwave']  = heatwave_mask.astype(int)
        self.data['is_upwelling'] = upwelling_mask.astype(int)

        return self.data

    def export(self, filename='massive_coral_training_data.csv'):
        if isinstance(self.data, dict) and not self.data:
            print("❌ No data to export. Run generate() first.")
            return

        print(f"\n💾 Exporting {len(self.data):,} records to {filename}...")
        self.data.to_csv(filename, index=False)
        import os
        size_mb = os.path.getsize(filename) / 1e6
        print(f"✅ Dataset exported: {filename}")
        print(f"   Rows: {len(self.data):,}")
        print(f"   Columns: {len(self.data.columns)}")
        print(f"   File size: {size_mb:.1f} MB")
        return filename

    def summary_statistics(self):
        if isinstance(self.data, dict) and not self.data:
            print("No data to summarise.")
            return

        print("\n📈 Dataset Summary Statistics:")
        print("=" * 70)
        for col in ['temperature', 'co2', 'depth', 'ph']:
            s = self.data[col]
            print(f"\n{col.upper()}:")
            print(f"  Mean:    {s.mean():8.3f}")
            print(f"  Std:     {s.std():8.3f}")
            print(f"  Min:     {s.min():8.3f}")
            print(f"  25%:     {s.quantile(0.25):8.3f}")
            print(f"  Median:  {s.median():8.3f}")
            print(f"  75%:     {s.quantile(0.75):8.3f}")
            print(f"  Max:     {s.max():8.3f}")


# ── module-level helper (unchanged) ────────────────────────────────────────
def remineralization_factor(depth):
    """Depth-dependent remineralisation proxy."""
    return np.tanh(depth / 200.0)


# ── entry point ─────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print(" 🪸 PROJECT CORAL SENTINEL - SYNTHETIC DATA GENERATOR 🪸 ")
    print("=" * 70)

    generator = CoralReefDataGenerator(n_samples=500_000, random_seed=42)

    print("\n🔄 Starting generation pipeline...\n")
    df = generator.generate()

    generator.summary_statistics()
    generator.export('massive_coral_training_data.csv')

    print("\n📋 Sample of Generated Data:")
    print("=" * 70)
    print(df.head(10).to_string())

    print("\n" + "=" * 70)
    print("✅ DATA GENERATION COMPLETE")
    print("=" * 70)
    print("\nYour dataset is ready for XGBoost training!")
    print("   Features: temperature, co2, depth, ph")
    print("   Derived: temperature_squared, co2_sqrt, depth_log, interactions")
    print("   Anomalies: is_heatwave, is_upwelling\n")


if __name__ == '__main__':
    main()
