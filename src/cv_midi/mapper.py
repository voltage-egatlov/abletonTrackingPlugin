"""Feature-to-MIDI CC mapper with smoothing and configurable ranges."""
from pydantic import BaseModel, field_validator


class MappingConfig(BaseModel):
    cc: int
    channel: int = 0
    min_val: int = 0
    max_val: int = 127
    smoothing: float = 0.0

    @field_validator("cc")
    @classmethod
    def validate_cc(cls, v):
        if not 0 <= v <= 127:
            raise ValueError(f"CC number must be 0-127, got {v}")
        return v

    @field_validator("channel")
    @classmethod
    def validate_channel(cls, v):
        if not 0 <= v <= 15:
            raise ValueError(f"MIDI channel must be 0-15, got {v}")
        return v

    @field_validator("smoothing")
    @classmethod
    def validate_smoothing(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Smoothing must be 0.0-1.0, got {v}")
        return v


class Mapper:
    """Maps named feature floats (0.0-1.0) to MIDI CC messages."""

    def __init__(self, config: dict[str, MappingConfig]):
        self._config = config
        self._smoothed: dict[str, float] = {}

    def process(self, features: dict[str, float]) -> list[tuple[int, int, int]]:
        """Return list of (channel, cc, value) tuples for each known feature."""
        messages = []
        for name, cfg in self._config.items():
            if name not in features:
                continue

            raw = max(0.0, min(1.0, features[name]))

            # Exponential moving average smoothing
            prev = self._smoothed.get(name, raw)
            smoothed = cfg.smoothing * prev + (1.0 - cfg.smoothing) * raw
            self._smoothed[name] = smoothed

            # Scale to MIDI range
            value = int(round(cfg.min_val + smoothed * (cfg.max_val - cfg.min_val)))
            value = max(0, min(127, value))
            messages.append((cfg.channel, cfg.cc, value))

        return messages
