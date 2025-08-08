# src/utils/price_dampening.py
import math

class PriceDampening:
    """
    Converts raw PPS (call it z or Δ_raw) into a dampened percent change.

    Implements the EXACT equations you provided:

    CONSERVATIVE
      Upside (Δ_raw >= 0):
        Tier 1 (0 <= z <= 2):
            Δ_damped = 1.0 * z / sqrt(1 + 4.0 * z^2)
        Tier 2 (z > 2):
            Δ_damped = 1.5 / (1 + exp(-6 * (z - 2)))

      Downside (Δ_raw < 0):
            Δ_damped = 1.0 * Δ_raw / sqrt(1 + 2.5 * Δ_raw^2)

    AGGRESSIVE
      Upside (z >= 0) uses a min() "cap" VALUE within the function:
        let base = z / sqrt(max(1 - 0.216 * z^2, 0.001))
        if 0.15 < z <= 0.5:
            Δ_final = 1.4 * min(base, CAP)
        else:
            Δ_final = min(base, CAP)
        where CAP = 20 (intragame) or 30 (weekly/monthly)

      Downside (z <= 0):
        Δ_final = -(|z|^2) / (|z|^2 + 0.18)

    Notes
    -----
    * All returned values are interpreted as "%", i.e., new_price = old_price * (1 + Δ/100).
    * All dampening functions return raw values that get divided by 100 in calculate_final_price.
    """

    # PPS weights used elsewhere (unchanged here, included for reference):
    # PTS 0.475, REB 0.15, AST 0.15, STOCKS 0.075, TO 0.05, FG% 0.10
    # (Weights are handled in the engine when computing PPS itself.)

    @staticmethod
    def _cap_for_timeframe(timeframe: str) -> float:
        """Return the CAP value used inside the aggressive upside function."""
        if timeframe == "intragame":
            return 20.0
        # weekly, monthly -> 30
        return 30.0

    @staticmethod
    def dampen_delta(raw_delta: float, strategy: str, timeframe: str) -> float:
        """
        Convert raw PPS into dampened Δ% based on strategy + timeframe.

        Args:
            raw_delta: PPS value (z, Δ_raw)
            strategy:  "conservative" or "aggressive"
            timeframe: "intragame" | "weekly" | "monthly"
        Returns:
            dampened percent change (float)
        """
        z = float(raw_delta)

        if strategy == "conservative":
            if z >= 0.0:
                # Upside: two tiers
                if z <= 2.0:
                    # Tier 1
                    denom = math.sqrt(1.0 + 4.0 * (z ** 2))
                    return (1.0 * z) / denom
                else:
                    # Tier 2
                    return 150 / (1.0 + math.exp(-6.0 * (z - 2.0)))
            else:
                # Downside
                denom = math.sqrt(1.0 + 2.5 * (z ** 2))
                return (1.0 * z) / denom

        # AGGRESSIVE
        if z >= 0.0:
            cap = PriceDampening._cap_for_timeframe(timeframe)
            # guard against negative inside sqrt
            inner = max(1.0 - 0.216 * (z ** 2), 0.001)
            base = z / math.sqrt(inner)

            if 0.15 < z <= 0.5:
                return min(1.4 * base, cap)
            else:
                return min(base, cap)
        else:
            az = abs(z)
            return -(az ** 2) / ((az ** 2) + 0.18)

    @staticmethod
    def calculate_final_price(old_price: float, dampened_delta: float) -> float:
        """Apply dampened percent change to get new price."""
        return float(old_price) * (1.0 + float(dampened_delta))
