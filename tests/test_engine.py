import pytest
import torch

from holos_tts_fastapi.config import Settings
from holos_tts_fastapi.engine import SynthesisEngine


class FakeModel:
    """Fails above a chosen speed, the way a chunk that rounds to zero frames does."""

    def __init__(self, fails_above: float):
        self.fails_above = fails_above
        self.speeds: list[float] = []

    def generate(self, text: str, voice: str, speed: float) -> torch.Tensor:
        self.speeds.append(speed)
        if speed > self.fails_above:
            raise RuntimeError("Kernel size can't be greater than actual input size")
        return torch.zeros(1, 2400)


def make_engine(model: FakeModel) -> SynthesisEngine:
    engine = SynthesisEngine(Settings())
    engine._model = model
    return engine


def test_chunk_too_short_for_high_speed_retries_at_normal_speed():
    model = FakeModel(fails_above=1.0)
    wav = make_engine(model)._generate_one("ʋ.", "voice", 4.0)
    assert wav.shape == (2400,)
    assert model.speeds == [4.0, 1.0]


def test_successful_chunk_keeps_requested_speed():
    model = FakeModel(fails_above=4.0)
    make_engine(model)._generate_one("ʋ.", "voice", 4.0)
    assert model.speeds == [4.0]


def test_failure_that_persists_at_normal_speed_propagates():
    model = FakeModel(fails_above=0.0)
    with pytest.raises(RuntimeError):
        make_engine(model)._generate_one("ʋ.", "voice", 4.0)
    assert model.speeds == [4.0, 1.0]


@pytest.mark.parametrize("speed", [0.25, 1.0])
def test_failure_at_or_below_normal_speed_is_not_retried(speed):
    model = FakeModel(fails_above=0.0)
    with pytest.raises(RuntimeError):
        make_engine(model)._generate_one("ʋ.", "voice", speed)
    assert model.speeds == [speed]
