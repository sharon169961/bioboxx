#!/usr/bin/env python3
"""
Project Coral Sentinel — ROS 2 Inference Node
Stateful pH prediction engine for live coral reef environmental data streams.

Subscriptions
-------------
  /raw_environment_data   std_msgs/Float32MultiArray
      Expects 7 floats per message:
      [node_id, temperature (°C), humidity (%), co2 (ppm), voc, pm25, depth (m)]
      (Indices 0, 1, 3, 6 are used; others are ignored.)

Publications
------------
  /coral_health/telemetry       std_msgs/String  (JSON payload)
  /coral_health/marker_color    std_msgs/ColorRGBA

Design
------
  • Maintains a rolling deque(maxlen=30) for temperature and CO2 to mirror the
    30-day windowed features engineered during training.
  • Derives all 19 features in the exact column order the StandardScaler was
    fitted on — any reordering would silently corrupt predictions.
  • Defers prediction until the buffer is full (30 readings).  During warm-up
    a WARNING is logged and no prediction is published.
  • Scaler and model are loaded once in __init__; inference is O(1) per tick.

Fixes (v2)
----------
  FIX 1 — Temporal Extrapolation Clamp (critical):
    The model was trained with year in [0, 4] (representing 2018–2022).
    datetime.now().year - 2018 currently yields 8, placing every prediction
    in an out-of-distribution leaf and artificially depressing pH regardless
    of sensor input.  The temporal features (year, month, day_of_year) are
    now clamped to a static pre-industrial baseline (year=0, month=6,
    day_of_year=180) so the model evaluates physical chemistry only.
    See: TEMPORAL_BASELINE_* constants and _engineer_features().

  FIX 2 — Message Layout Index Correction (critical):
    test_sensor_publisher.py publishes a 7-element array:
      [node_id, temp, humidity, co2, voc, pm25, depth]
    The original node incorrectly parsed raw[0,1,2] as (temp, co2, depth),
    which mapped node_id→temperature, temp→CO2, and humidity→depth.
    Correct indices are now: temp=raw[1], co2=raw[3], depth=raw[6].
"""

import json
import os
import traceback
from collections import deque
from datetime import datetime, timezone

import joblib
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, Float32MultiArray, String
import xgboost as xgb

# ── constants ─────────────────────────────────────────────────────────────────

# Exact column order the StandardScaler was fitted on in feature_engineer.py.
# DO NOT reorder — scaler.transform() is positionally sensitive.
FEATURE_COLS: list[str] = [
    "temperature",
    "co2",
    "depth",
    "temperature_squared",
    "co2_sqrt",
    "depth_log",
    "temp_depth_interaction",
    "co2_temp_ratio",
    "day_of_year",
    "year",
    "month",
    "is_heatwave",
    "is_upwelling",
    "temp_rolling_7d",
    "co2_rolling_7d",
    "temp_rolling_30d",
    "co2_rolling_30d",
    "temp_diff_24h",
    "co2_diff_24h",
]

# ── FIX 1: Temporal baseline clamp ───────────────────────────────────────────
# The XGBoost model was trained on data_generator.py output where:
#   year        ∈ [0, 4]   (0 = 2018, 4 = 2022)
#   month       ∈ [1, 12]
#   day_of_year ∈ [1, 365]
#
# In production (2026) datetime.now().year - 2018 = 8, which is OOB and forces
# every reading into the model's most extreme "late year" leaf node, depressing
# the pH prediction regardless of the physical sensor chemistry.
#
# Resolution: freeze temporal features to a neutral mid-year baseline inside
# the training distribution.  year=0 (2018) is the safest anchor — it sits at
# the low-CO2 end of the training range and does not impose any seasonal bias
# beyond what month/day_of_year provide.
#
# If you retrain the model with a year range that covers the deployment period,
# remove these constants and restore the live datetime calculation.
TEMPORAL_BASELINE_YEAR        = 0    # Equivalent to calendar year 2018
TEMPORAL_BASELINE_MONTH       = 6    # Mid-year; neutral seasonal position
TEMPORAL_BASELINE_DAY_OF_YEAR = 180  # ~June 29; mid-year, avoids seasonal edges
# ─────────────────────────────────────────────────────────────────────────────

# Buffer size must cover the longest rolling window (30 days)
BUFFER_MAXLEN: int = 30
ROLL_SHORT:    int = 7
ROLL_LONG:     int = 30

# ── FIX 2: Message layout indices ────────────────────────────────────────────
# Publisher array: [node_id, temp, humidity, co2, voc, pm25, depth]
MSG_IDX_NODE_ID     = 0
MSG_IDX_TEMPERATURE = 1
MSG_IDX_CO2         = 3
MSG_IDX_DEPTH       = 6
MSG_MIN_LEN         = 7   # must have at least 7 elements
# ─────────────────────────────────────────────────────────────────────────────

# Anomaly thresholds (match training data_generator.py logic)
HEATWAVE_TEMP_THRESHOLD: float = 30.0   # °C
UPWELLING_TEMP_DROP:     float = -1.5   # °C per tick (1-step diff)
UPWELLING_CO2_SPIKE:     float = 10.0   # ppm per tick

# pH → acidification risk colour map
# Green (healthy) → Yellow (stressed) → Red (critical)
_PH_COLOUR_MAP = [
    (8.20, (0.0,  0.85, 0.2,  1.0)),   # healthy     — green
    (8.05, (0.6,  0.85, 0.0,  1.0)),   # mild stress — yellow-green
    (7.90, (1.0,  0.75, 0.0,  1.0)),   # moderate    — amber
    (7.75, (1.0,  0.35, 0.0,  1.0)),   # high stress — orange
    (0.00, (0.9,  0.0,  0.0,  1.0)),   # critical    — red
]

# Artifact paths — override via ROS 2 parameters if needed
DEFAULT_SCALER_PATH = "advanced_scaler.pkl"
DEFAULT_MODEL_PATH  = "advanced_coral_model.json"


# ── node ──────────────────────────────────────────────────────────────────────

class CoralInferenceNode(Node):
    """
    Stateful ROS 2 node that ingests raw environmental sensor data,
    reconstructs training-time time-series features, and publishes
    real-time pH predictions and acidification risk colour markers.
    """

    def __init__(self) -> None:
        super().__init__("coral_inference_node")

        # ── ROS 2 parameters (paths overridable at launch time) ──────────
        self.declare_parameter("scaler_path", DEFAULT_SCALER_PATH)
        self.declare_parameter("model_path",  DEFAULT_MODEL_PATH)

        scaler_path = self.get_parameter("scaler_path").get_parameter_value().string_value
        model_path  = self.get_parameter("model_path").get_parameter_value().string_value

        # ── load artifacts ───────────────────────────────────────────────
        self._scaler = self._load_scaler(scaler_path)
        self._model  = self._load_model(model_path)

        # Validate scaler expects exactly the right number of features
        if hasattr(self._scaler, "n_features_in_"):
            expected = self._scaler.n_features_in_
            actual   = len(FEATURE_COLS)
            if expected != actual:
                raise RuntimeError(
                    f"Scaler expects {expected} features but node defines "
                    f"{actual}. Verify FEATURE_COLS matches feature_engineer.py."
                )

        # ── state: rolling history buffers ───────────────────────────────
        # deque automatically discards readings older than BUFFER_MAXLEN days
        self._temp_buf:   deque[float] = deque(maxlen=BUFFER_MAXLEN)
        self._co2_buf:    deque[float] = deque(maxlen=BUFFER_MAXLEN)
        self._tick_count: int = 0   # total messages received (for logging)

        # ── publishers ───────────────────────────────────────────────────
        self._telemetry_pub = self.create_publisher(
            String, "/coral_health/telemetry", qos_profile=10
        )
        self._colour_pub = self.create_publisher(
            ColorRGBA, "/coral_health/marker_color", qos_profile=10
        )

        # ── subscriber ───────────────────────────────────────────────────
        self._data_sub = self.create_subscription(
            Float32MultiArray,
            "/raw_environment_data",
            self._on_raw_data,
            qos_profile=10,
        )

        self.get_logger().info(
            "CoralInferenceNode ready — waiting for data on "
            "/raw_environment_data  (warm-up: %d readings needed)",
            BUFFER_MAXLEN,
        )
        self.get_logger().info(
            "Temporal baseline active: year=%d  month=%d  day_of_year=%d  "
            "(OOB extrapolation guard — retrain model to lift this clamp)",
            TEMPORAL_BASELINE_YEAR,
            TEMPORAL_BASELINE_MONTH,
            TEMPORAL_BASELINE_DAY_OF_YEAR,
        )

    # ── artifact loaders ──────────────────────────────────────────────────────

    def _load_scaler(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Scaler artifact not found: '{path}'. "
                "Run feature_engineer.py first."
            )
        scaler = joblib.load(path)
        self.get_logger().info("Scaler loaded: %s", path)
        return scaler

    def _load_model(self, path: str) -> xgb.XGBRegressor:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Model artifact not found: '{path}'. "
                "Run train_xgboost_optimized.py first."
            )
        model = xgb.XGBRegressor()
        model.load_model(path)       # native JSON load — no pickle, version-safe
        self.get_logger().info("XGBoost model loaded: %s", path)
        return model

    # ── main callback ─────────────────────────────────────────────────────────

    def _on_raw_data(self, msg: Float32MultiArray) -> None:
        """
        Called on every incoming sensor reading.

        Expected msg.data layout (7 elements):
          [node_id, temperature, humidity, co2, voc, pm25, depth]
          Indices used: 1 (temp), 3 (co2), 6 (depth).
        """
        self._tick_count += 1

        # ── parse and validate incoming floats ───────────────────────────
        try:
            raw = list(msg.data)

            # FIX 2: require full 7-element message; old check was len < 3
            # which silently accepted messages and read wrong indices.
            if len(raw) < MSG_MIN_LEN:
                self.get_logger().error(
                    "Malformed message: expected %d floats "
                    "[node_id, temp, humidity, co2, voc, pm25, depth], "
                    "got %d. Skipping.",
                    MSG_MIN_LEN, len(raw),
                )
                return

            node_id     = int(raw[MSG_IDX_NODE_ID])
            temperature = float(raw[MSG_IDX_TEMPERATURE])
            co2         = float(raw[MSG_IDX_CO2])
            depth       = float(raw[MSG_IDX_DEPTH])

            # Sanity-check physical ranges (mirrors training data clipping)
            if not (-2.0 <= temperature <= 35.0):
                self.get_logger().warning(
                    "Temperature %.2f°C out of expected range [-2, 35]. "
                    "Clamping.", temperature
                )
                temperature = float(np.clip(temperature, -2.0, 35.0))

            if not (200.0 <= co2 <= 800.0):
                self.get_logger().warning(
                    "CO2 %.1f ppm out of expected range [200, 800]. "
                    "Clamping.", co2
                )
                co2 = float(np.clip(co2, 200.0, 800.0))

            if not (0.5 <= depth <= 3000.0):
                self.get_logger().warning(
                    "Depth %.1f m out of expected range [0.5, 3000]. "
                    "Clamping.", depth
                )
                depth = float(np.clip(depth, 0.5, 3000.0))

        except (TypeError, ValueError) as exc:
            self.get_logger().error("Failed to parse message data: %s", exc)
            return

        # ── update state buffers ─────────────────────────────────────────
        # Append AFTER parsing so a bad message never contaminates the buffer.
        self._temp_buf.append(temperature)
        self._co2_buf.append(co2)

        buf_len = len(self._temp_buf)

        # ── warm-up gate ─────────────────────────────────────────────────
        # Prediction requires BUFFER_MAXLEN readings so all rolling features
        # are available — exactly mirroring the dropna() in feature_engineer.py.
        if buf_len < BUFFER_MAXLEN:
            self.get_logger().warning(
                "Warm-up: %d / %d readings collected. "
                "Prediction deferred until buffer is full.",
                buf_len, BUFFER_MAXLEN,
            )
            return

        # ── feature engineering ──────────────────────────────────────────
        try:
            features = self._engineer_features(temperature, co2, depth)
        except Exception as exc:
            self.get_logger().error(
                "Feature engineering failed: %s\n%s",
                exc, traceback.format_exc()
            )
            return

        # ── scale & predict ──────────────────────────────────────────────
        try:
            predicted_ph = self._predict(features)
        except Exception as exc:
            self.get_logger().error(
                "Prediction failed: %s\n%s", exc, traceback.format_exc()
            )
            return

        # ── publish outputs ──────────────────────────────────────────────
        self._publish_telemetry(
            node_id, temperature, co2, depth, features, predicted_ph
        )
        self._publish_colour(predicted_ph)

        self.get_logger().info(
            "[tick %d | node %d]  T=%.2f°C  CO2=%.1fppm  depth=%.1fm  "
            "→  predicted pH=%.4f  (%s)  heatwave=%d  upwelling=%d",
            self._tick_count,
            node_id,
            temperature, co2, depth,
            predicted_ph,
            self._ph_to_risk(predicted_ph)[0],
            int(features["is_heatwave"]),
            int(features["is_upwelling"]),
        )

    # ── feature engineering ───────────────────────────────────────────────────

    def _engineer_features(self, temperature: float, co2: float, depth: float) -> dict:
        """
        Compute all 19 training-time features from live sensor values and
        the current state of the history buffers.

        Returns a dict keyed by FEATURE_COLS in order (order is validated
        downstream before scaler.transform is called).
        """
        # ── rolling statistics ───────────────────────────────────────────
        # Buffers are already updated before this method is called.
        temp_list = list(self._temp_buf)   # snapshot; deque may keep updating
        co2_list  = list(self._co2_buf)

        temp_rolling_7d  = float(np.mean(temp_list[-ROLL_SHORT:]))
        co2_rolling_7d   = float(np.mean(co2_list[-ROLL_SHORT:]))
        temp_rolling_30d = float(np.mean(temp_list[-ROLL_LONG:]))
        co2_rolling_30d  = float(np.mean(co2_list[-ROLL_LONG:]))

        # 1-step diff mirrors pandas .diff() — current minus previous reading
        temp_diff_24h = float(temp_list[-1] - temp_list[-2])
        co2_diff_24h  = float(co2_list[-1]  - co2_list[-2])

        # ── FIX 1: Temporal features — clamped to training baseline ──────
        #
        # REMOVED:
        #   now         = datetime.now(tz=timezone.utc)
        #   day_of_year = int(now.strftime("%j"))       # → up to 365 (fine)
        #   year_offset = now.year - 2018               # → 8 in 2026 (OOB!)
        #   month       = now.month                     # → fine on its own
        #
        # REPLACED WITH: static constants that sit safely inside [0, 4].
        # This removes calendar-year influence entirely and lets the model
        # evaluate the physical chemistry (temperature, CO2) without the
        # year feature dragging it toward its extreme acidification leaf.
        #
        # To restore live temporal features, retrain data_generator.py with
        # a year range that includes the deployment era (e.g. year 0..8) and
        # then replace these three lines with the datetime block above.
        day_of_year = TEMPORAL_BASELINE_DAY_OF_YEAR
        year_offset = TEMPORAL_BASELINE_YEAR
        month       = TEMPORAL_BASELINE_MONTH
        # ─────────────────────────────────────────────────────────────────

        # ── anomaly flags ────────────────────────────────────────────────
        # is_heatwave: temperature exceeds thermal stress threshold
        is_heatwave = int(temperature > HEATWAVE_TEMP_THRESHOLD)

        # is_upwelling: simultaneous sharp temp drop AND CO2 spike
        # (cold, CO2-rich deep water surfacing — matches training generator logic)
        is_upwelling = int(
            temp_diff_24h < UPWELLING_TEMP_DROP and
            co2_diff_24h  > UPWELLING_CO2_SPIKE
        )

        # ── derived scalar features ──────────────────────────────────────
        temperature_squared    = temperature ** 2
        co2_sqrt               = float(np.sqrt(max(co2, 0.0)))   # safe: co2 ≥ 200 post-clip
        depth_log              = float(np.log1p(depth))
        temp_depth_interaction = temperature * depth_log
        co2_temp_ratio         = co2 / (temperature + 1e-6)      # matches training guard

        # ── assemble in EXACT training column order ───────────────────────
        return {
            "temperature"           : temperature,
            "co2"                   : co2,
            "depth"                 : depth,
            "temperature_squared"   : temperature_squared,
            "co2_sqrt"              : co2_sqrt,
            "depth_log"             : depth_log,
            "temp_depth_interaction": temp_depth_interaction,
            "co2_temp_ratio"        : co2_temp_ratio,
            "day_of_year"           : day_of_year,
            "year"                  : year_offset,
            "month"                 : month,
            "is_heatwave"           : is_heatwave,
            "is_upwelling"          : is_upwelling,
            "temp_rolling_7d"       : temp_rolling_7d,
            "co2_rolling_7d"        : co2_rolling_7d,
            "temp_rolling_30d"      : temp_rolling_30d,
            "co2_rolling_30d"       : co2_rolling_30d,
            "temp_diff_24h"         : temp_diff_24h,
            "co2_diff_24h"          : co2_diff_24h,
        }

    # ── scaler + model ────────────────────────────────────────────────────────

    def _predict(self, features: dict) -> float:
        """
        Scale the feature vector and run XGBoost inference.

        The feature dict is converted to a numpy row in FEATURE_COLS order
        before scaling — this is the critical alignment step.
        """
        # Build row in the exact column order the scaler was fitted on
        row = np.array(
            [features[col] for col in FEATURE_COLS],
            dtype=np.float64
        ).reshape(1, -1)

        # Detect and refuse NaN/Inf before they corrupt predictions silently
        if not np.isfinite(row).all():
            bad = [FEATURE_COLS[i] for i in range(len(FEATURE_COLS))
                   if not np.isfinite(row[0, i])]
            raise ValueError(f"Non-finite values in feature vector: {bad}")

        scaled_row   = self._scaler.transform(row)           # (1, 19) float64
        predicted_ph = float(self._model.predict(scaled_row)[0])

        # Final physical sanity clip (model won't predict outside [6.5, 9.0]
        # for real ocean data, but guard against extrapolation artefacts)
        predicted_ph = float(np.clip(predicted_ph, 6.5, 9.0))
        return predicted_ph

    # ── publishers ────────────────────────────────────────────────────────────

    def _publish_telemetry(
        self,
        node_id: int,
        temperature: float,
        co2: float,
        depth: float,
        features: dict,
        predicted_ph: float,
    ) -> None:
        """Publish a rich JSON telemetry payload to /coral_health/telemetry."""
        risk_label, risk_score = self._ph_to_risk(predicted_ph)

        payload = {
            "timestamp_utc"  : datetime.now(tz=timezone.utc).isoformat(),
            "tick"           : self._tick_count,
            "node_id"        : node_id,
            # raw sensor values
            "sensor": {
                "temperature_c" : round(temperature, 4),
                "co2_ppm"       : round(co2, 4),
                "depth_m"       : round(depth, 4),
            },
            # prediction
            "prediction": {
                "ph"            : round(predicted_ph, 6),
                "risk_label"    : risk_label,
                "risk_score"    : risk_score,
            },
            # time-series context (useful for downstream dashboards)
            "rolling": {
                "temp_7d_mean"  : round(features["temp_rolling_7d"],  4),
                "temp_30d_mean" : round(features["temp_rolling_30d"], 4),
                "co2_7d_mean"   : round(features["co2_rolling_7d"],   4),
                "co2_30d_mean"  : round(features["co2_rolling_30d"],  4),
                "temp_diff_24h" : round(features["temp_diff_24h"],    4),
                "co2_diff_24h"  : round(features["co2_diff_24h"],     4),
            },
            # anomaly flags
            "anomalies": {
                "is_heatwave"   : bool(features["is_heatwave"]),
                "is_upwelling"  : bool(features["is_upwelling"]),
            },
            # diagnostic: confirm temporal clamp is active
            "temporal_baseline": {
                "year"          : features["year"],
                "month"         : features["month"],
                "day_of_year"   : features["day_of_year"],
                "clamped"       : True,
            },
        }

        msg = String()
        msg.data = json.dumps(payload)
        self._telemetry_pub.publish(msg)

    def _publish_colour(self, predicted_ph: float) -> None:
        """
        Map predicted pH to a ColorRGBA marker.

        pH ≥ 8.20  →  green   (healthy)
        pH ≥ 8.05  →  yellow-green (mild stress)
        pH ≥ 7.90  →  amber   (moderate stress)
        pH ≥ 7.75  →  orange  (high stress)
        pH <  7.75 →  red     (critical acidification)
        """
        r, g, b, a = self._ph_to_rgba(predicted_ph)

        msg   = ColorRGBA()
        msg.r = float(r)
        msg.g = float(g)
        msg.b = float(b)
        msg.a = float(a)
        self._colour_pub.publish(msg)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ph_to_rgba(ph: float) -> tuple[float, float, float, float]:
        """Return (r, g, b, a) for a given pH using the colour map."""
        for threshold, colour in _PH_COLOUR_MAP:
            if ph >= threshold:
                return colour
        return _PH_COLOUR_MAP[-1][1]   # red fallback (should never reach here)

    @staticmethod
    def _ph_to_risk(ph: float) -> tuple[str, int]:
        """Return (human-readable label, integer risk score 0–4)."""
        if ph >= 8.20:
            return ("HEALTHY",  0)
        elif ph >= 8.05:
            return ("MILD",     1)
        elif ph >= 7.90:
            return ("MODERATE", 2)
        elif ph >= 7.75:
            return ("HIGH",     3)
        else:
            return ("CRITICAL", 4)


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)

    try:
        node = CoralInferenceNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        # Use Python logging as fallback if ROS logger is unavailable
        import logging
        logging.basicConfig()
        logging.getLogger(__name__).critical(
            "Fatal error in CoralInferenceNode: %s\n%s",
            exc, traceback.format_exc()
        )
        raise
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
