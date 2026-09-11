"""Measure paired soft/hard DDM discrepancies on seeded simulated paths."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import tyro

from agents.ddm_simulation import soft_first_passage


@dataclass(slots=True)
class Args:
    """Numerical calibration settings; these are not animal experiments."""

    output: Path = Path("docs/results/ddm_surrogate_calibration.json")
    samples: int = 3000
    seed: int = 81


def main(args: Args) -> None:
    """Write reproducible paired numerical comparisons across temperatures."""
    rows = []
    for drift, bound, noise in [(-1., .5, 1.), (0., 1., 1.), (2., 1.5, .5)]:
        generator = torch.Generator().manual_seed(args.seed)
        paths = (drift * .01 + noise * .1 * torch.randn(args.samples, 150, generator=generator)).cumsum(-1)
        crosses = paths.abs() >= bound
        first = crosses.to(torch.int64).argmax(-1)
        committed = crosses.any(-1)
        indices = torch.where(committed, first, torch.full_like(first, 149))
        final = paths.gather(-1, indices[:, None]).squeeze(-1)
        hard_p = (final > 0).float().mean().item()
        hard_rt = ((indices + 1 + 15).clamp(5, 150) * 10).float().mean().item()
        for temperature in [.1, .02, .01, .005, .001]:
            p, rt = soft_first_passage(paths, torch.tensor(bound), torch.tensor(150.),
                                       step_ms=10, min_commit_steps=5, max_commit_steps=150,
                                       temperature=temperature)
            rows.append(dict(drift=drift, half_bound=bound, noise=noise,
                             temperature=temperature, hard_p_right=hard_p,
                             soft_p_right=p.mean().item(), p_right_difference=p.mean().item()-hard_p,
                             hard_mean_rt_ms=hard_rt, soft_mean_rt_ms=rt.mean().item(),
                             mean_rt_difference_ms=rt.mean().item()-hard_rt))
    payload = dict(settings={**asdict(args), "output": str(args.output)},
                   step_ms=10, maximum_steps=150, minimum_steps=5, non_decision_ms=150,
                   lapse_rate=0, initial_position=0, torch_version=torch.__version__,
                   interpretation="Paired numerical calibration on three simulated regimes; not animal validation or an exhaustive error bound.",
                   limitations=["Soft hazards differ from hard crossing at any nonzero temperature.",
                                "Narrow temperatures can increase gradient variance or saturation.",
                                "Exact-boundary ties have sigmoid hazard one half.",
                                "Quantized RT uses a straight-through gradient."], rows=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(row['drift'], row['half_bound'], row['temperature'],
              round(row['p_right_difference'], 5), round(row['mean_rt_difference_ms'], 3))


if __name__ == "__main__":
    main(tyro.cli(Args))
