"""
Evaluation-time scheduling for the forward model.

Observations carry calendar-year timestamps, and the forward model must emit
state exactly at those times. `build_step_sequence` designs the sequence of
(time, dt) steps: a uniform spinup grid at the configured dt, with extra
breakpoints snapped onto every required observation time (variable dt, capped
at the configured maximum). Times are plain Python floats throughout so
snapshot dictionary keys match the requested times exactly — no float32
clock-drift matching.
"""
from typing import Iterable, Optional, Sequence


def merge_times(*iterables: Optional[Iterable[float]], eps: float = 1e-6
                ) -> tuple:
    """Union of time values from any number of (possibly None) iterables,
    sorted ascending and deduplicated within `eps`."""
    times: list = []
    for it in iterables:
        if it is None:
            continue
        times.extend(float(t) for t in it)
    times.sort()
    merged: list = []
    for t in times:
        if not merged or t - merged[-1] > eps:
            merged.append(t)
    return tuple(merged)


def build_step_sequence(
    *,
    t_start: float,
    t_end: float,
    dt_max: float,
    required_times: Sequence[float] = (),
    eps: float = 1e-6,
) -> list:
    """Design the forward model's step sequence as [(t_next, dt), ...].

    The run covers (t_start, horizon] where horizon = max(t_end, max required
    time) — a required time beyond the nominal calibration end extends the run
    (climate-anomaly lookups clamp to the last available year, so forcing is
    defined). Breakpoints are the uniform grid {t_start + k * dt_max} unioned
    with `required_times`; a grid point within `eps` of a required time is
    replaced by the required time so state is emitted at exactly the requested
    float. Every dt is positive and <= dt_max + eps.

    Raises ValueError if a required time falls at or before t_start (the model
    cannot emit state it never simulates — lower t_start or override the
    observation time), or if the sequence would be empty.
    """
    t_start = float(t_start)
    t_end = float(t_end)
    dt_max = float(dt_max)
    if dt_max <= 0:
        raise ValueError(f"dt_max must be positive, got {dt_max}")

    required = merge_times(required_times, eps=eps)
    too_early = [t for t in required if t <= t_start + eps]
    if too_early:
        raise ValueError(
            f"required time(s) {too_early} fall at or before t_start="
            f"{t_start}; lower t_start (or override the observation time) so "
            f"the model simulates through every observation epoch."
        )

    horizon = max([t_end] + list(required))

    # Uniform grid, skipping points that collide (within eps) with a required
    # time — the required float wins so snapshot keys are exact.
    breakpoints = list(required)
    k = 1
    while True:
        t = t_start + k * dt_max
        if t > horizon - eps:
            break
        if all(abs(t - r) > eps for r in required):
            breakpoints.append(t)
        k += 1
    if all(abs(horizon - b) > eps for b in breakpoints):
        breakpoints.append(horizon)
    breakpoints.sort()

    steps = []
    t_prev = t_start
    for t_next in breakpoints:
        steps.append((t_next, t_next - t_prev))
        t_prev = t_next
    if not steps:
        raise ValueError(
            f"empty step sequence for t_start={t_start}, t_end={t_end}, "
            f"required_times={required}; the horizon must exceed t_start."
        )
    return steps
