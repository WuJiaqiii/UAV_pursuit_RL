"""Low-overhead PPO progress display for terminals and redirected logs."""
import sys
import time


def _duration(seconds):
    if seconds is None or seconds < 0:
        return "--:--"
    value = int(seconds)
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _steps(value):
    value = float(value)
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(int(value))


def _rate(value):
    if value is None or value <= 0:
        return "--"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.1f}"


class TrainingProgress:
    """Render a tqdm-like bar on TTY, concise periodic lines in log files.

    The counter uses the trainer's aggregate physics-step budget (the exact
    quantity used by the training loop's stopping condition), not PPO updates.
    No CUDA tensor reads or synchronization are performed here.
    """

    def __init__(
        self,
        total_steps,
        initial_steps=0,
        initial_updates=0,
        completed_episodes=0,
        successful_episodes=0,
        stream=None,
        refresh_interval=0.5,
        log_interval=10.0,
        width=22,
        clock=None,
        enabled=True,
    ):
        self.total_steps = max(int(total_steps), 1)
        self.steps = int(initial_steps)
        self.initial_steps = int(initial_steps)
        self.updates = int(initial_updates)
        self.completed = int(completed_episodes)
        self.successful = int(successful_episodes)
        self.stream = stream if stream is not None else sys.stderr
        self.refresh_interval = max(float(refresh_interval), 0.0)
        self.log_interval = max(float(log_interval), 0.0)
        self.width = max(int(width), 8)
        self.clock = clock or time.monotonic
        self.enabled = bool(enabled)
        self.is_tty = bool(
            self.enabled
            and hasattr(self.stream, "isatty")
            and self.stream.isatty()
        )
        now = self.clock()
        self.started_at = now
        self.last_output_at = now
        self.last_line_width = 0
        self.phase = "training"

    def _line(self, now, status=None):
        fraction = min(max(self.steps / self.total_steps, 0.0), 1.0)
        filled = int(round(self.width * fraction))
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = max(now - self.started_at, 0.0)
        # Effective throughput includes optimizer and validation time, so
        # ETA reflects the full wall-clock training process, not rollout-only
        # simulator throughput.
        advanced_steps = max(self.steps - self.initial_steps, 0)
        rate = advanced_steps / elapsed if elapsed > 0 and advanced_steps > 0 else None
        remaining = max(self.total_steps - self.steps, 0)
        eta = remaining / rate if rate and rate > 0 else None
        success = (
            f"{100.0 * self.successful / self.completed:.1f}%"
            if self.completed else "n/a"
        )
        phase = status or self.phase
        return (
            f"Physics [{bar}] {100.0 * fraction:5.1f}% "
            f"{_steps(self.steps)}/{_steps(self.total_steps)} | "
            f"PPO {self.updates} | {_rate(rate)} steps/s | "
            f"Elapsed {_duration(elapsed)} | ETA {_duration(eta)} | "
            f"Train SR {success} ({self.completed} eps) | {phase}"
        )

    def update(
        self,
        steps,
        updates,
        completed_episodes,
        successful_episodes,
        phase="training",
        force=False,
    ):
        if not self.enabled:
            return
        now = self.clock()
        steps = int(steps)
        self.steps = steps
        self.updates = int(updates)
        self.completed = int(completed_episodes)
        self.successful = int(successful_episodes)
        self.phase = str(phase)

        if self.is_tty:
            if force or now - self.last_output_at >= self.refresh_interval:
                line = self._line(now)
                self.stream.write("\r" + line.ljust(self.last_line_width))
                self.stream.flush()
                self.last_line_width = len(line)
                self.last_output_at = now
        elif force or now - self.last_output_at >= self.log_interval:
            self.stream.write("[progress] " + self._line(now) + "\n")
            self.stream.flush()
            self.last_output_at = now

    def finish(self, status="completed"):
        if not self.enabled:
            return
        now = self.clock()
        line = self._line(now, status=status)
        if self.is_tty:
            self.stream.write("\r" + line.ljust(self.last_line_width) + "\n")
        else:
            self.stream.write("[progress] " + line + "\n")
        self.stream.flush()
        self.last_output_at = now
