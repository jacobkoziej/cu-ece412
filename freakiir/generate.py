#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
#
# generate.py -- generate points on the Riemann sphere
# Copyright (C) 2024  Jacob Koziej <jacobkoziej@gmail.com>
# Copyright (C) 2024  Eric Eng <eric.eng@cooper.edu>


def main() -> None:
    import numpy as np
    import pandas as pd

    from argparse import ArgumentParser
    from pathlib import Path

    argparser = ArgumentParser(
        description="generate uniform points on the unit sphere",
    )

    argparser.add_argument(
        "-e",
        "--epsilon",
        default=1e-8,
        help="pole distance from unit circle [rad]",
        type=float,
    )
    argparser.add_argument(
        "-n",
        "--points",
        help="number of points to generate",
        required=True,
        type=int,
    )
    argparser.add_argument(
        "-o",
        "--output",
        help="output name",
        required=True,
        type=Path,
    )
    argparser.add_argument(
        "-s",
        "--seed",
        default=0x851F3CE1,
        help="rng seed",
        type=int,
    )

    args = argparser.parse_args()

    theta_limits = (-np.pi, np.pi - np.finfo(np.pi).eps)
    zero_phi_limits = (0, np.pi)
    pole_phi_limits = ((np.pi / 2) + abs(args.epsilon), np.pi)

    rng = np.random.default_rng(args.seed)

    data = {
        "zero_0_theta": rng.uniform(*theta_limits, args.points),
        "zero_0_phi": rng.uniform(*zero_phi_limits, args.points),
        "zero_1_theta": rng.uniform(*theta_limits, args.points),
        "zero_1_phi": rng.uniform(*zero_phi_limits, args.points),
        "pole_0_theta": rng.uniform(*theta_limits, args.points),
        "pole_0_phi": rng.uniform(*pole_phi_limits, args.points),
        "pole_1_theta": rng.uniform(*theta_limits, args.points),
        "pole_1_phi": rng.uniform(*pole_phi_limits, args.points),
    }

    df = pd.DataFrame(data)

    df.to_csv(f"{args.output}.csv", index=False)

    with open(f"{args.output}.args", "w") as f:
        f.write(f"{args}\n")


if __name__ == "__main__":
    main()
