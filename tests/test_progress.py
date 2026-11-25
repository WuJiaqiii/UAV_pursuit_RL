"""TTY and redirected-log behavior for the PPO training progress display."""
import io

from pursuit_rl.ctbr.progress import TrainingProgress, _duration


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class FakeStream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


def test_tty_progress_uses_carriage_return_and_finishes_cleanly():
    clock = FakeClock()
    stream = FakeStream(tty=True)
    display = TrainingProgress(
        total_steps=100, stream=stream, clock=clock,
        refresh_interval=0, width=10,
    )
    display.update(0, 0, 0, 0, phase="starting", force=True)
    clock.advance(10)
    display.update(50, 5, 4, 3, phase="training", force=True)
    clock.advance(5)
    display.update(100, 10, 10, 8, phase="training", force=True)
    display.finish("completed")

    output = stream.getvalue()
    assert "\r" in output
    assert "100.0%" in output
    assert "100/100" in output
    assert "PPO 10" in output
    assert "Train SR 80.0% (10 eps)" in output
    assert output.endswith("\n")


def test_redirected_log_is_periodic_not_carriage_return_spam():
    clock = FakeClock()
    stream = FakeStream(tty=False)
    display = TrainingProgress(
        total_steps=100, stream=stream, clock=clock,
        log_interval=10, width=10,
    )
    display.update(0, 0, 0, 0, phase="starting", force=True)
    clock.advance(5)
    display.update(40, 2, 2, 1)
    assert stream.getvalue().count("[progress]") == 1
    clock.advance(6)
    display.update(60, 3, 4, 2, phase="training")
    assert stream.getvalue().count("[progress]") == 2
    clock.advance(1)
    display.finish("completed")

    output = stream.getvalue()
    assert "\r" not in output
    assert "60.0%" in output
    assert "ETA" in output
    assert "Train SR 50.0% (4 eps)" in output
    assert "completed" in output


def test_duration_formatting():
    assert _duration(None) == "--:--"
    assert _duration(65) == "01:05"
    assert _duration(3661) == "01:01:01"


def test_disabled_progress_writes_nothing():
    clock = FakeClock()
    stream = FakeStream(tty=False)
    display = TrainingProgress(
        total_steps=100, stream=stream, clock=clock, enabled=False,
    )
    display.update(50, 1, 0, 0, force=True)
    display.finish()
    assert stream.getvalue() == ""
