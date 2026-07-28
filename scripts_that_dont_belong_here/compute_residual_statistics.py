import argparse
import os
from multiprocessing import Pool
from typing import Iterable

import numpy as np
import tqdm
from anemoi.datasets import open_dataset
from anemoi.utils.text import table
from scipy.sparse import load_npz

FORCING_VARIABLES = set(
    [
        "cos_julian_day",
        "cos_latitude",
        "cos_local_time",
        "cos_longitude",
        "insolation",
        "lsm",
        "sin_julian_day",
        "sin_latitude",
        "sin_local_time",
        "sin_longitude",
        "z",
    ]
)


def update_running_stats(stats, residual, total_steps, chunk_size=1):
    """Update mean, stdev, min, max stats on the fly."""
    stats["mean"] += chunk_size * np.mean(residual) / total_steps
    stats["mean2"] += chunk_size * np.mean(residual**2) / total_steps
    stats["minimum"] = min(stats["minimum"], residual.min())
    stats["maximum"] = max(stats["maximum"], residual.max())


def finalize_stats(running_stats):
    """Convert accumulated stats to final mean/stdev."""
    results = {"mean": {}, "stdev": {}, "minimum": {}, "maximum": {}}

    for var, stats in running_stats.items():
        results["mean"][var] = stats["mean"]
        results["stdev"][var] = np.sqrt(stats["mean2"] - stats["mean"] ** 2)
        results["minimum"][var] = stats["minimum"]
        results["maximum"][var] = stats["maximum"]

    return results


def compute_variable_stats(arguments: tuple):
    (variable, hres_indices, lres_indices, args) = arguments
    total_steps = len(hres_indices)

    hres_dataset = open_dataset(args.hres_path)
    lres_dataset = open_dataset(args.lres_path)
    interp_matrix = load_npz(args.interp_matrix_path)

    stats = {"mean": 0.0, "mean2": 0.0, "minimum": np.inf, "maximum": -np.inf}

    hres_id = hres_dataset.name_to_index[variable]
    lres_id = lres_dataset.name_to_index[variable]

    for i in range(0, total_steps, args.chunk_size):
        h_chunk = hres_indices[i : i + args.chunk_size]
        l_chunk = lres_indices[i : i + args.chunk_size]
        actual_chunk_size = len(h_chunk)

        try:
            # (chunk, hres_size)
            hres_chunk = hres_dataset[h_chunk, hres_id, 0, :]

            # (chunk, lres_size)
            lres_chunk = lres_dataset[l_chunk, lres_id, 0, :]

            # Interpolation   # (chunk, lres_hres_sizesize)
            lres_chunk_interpolated = (interp_matrix @ lres_chunk.T).T
            residual = hres_chunk - lres_chunk_interpolated
            update_running_stats(stats, residual, total_steps, actual_chunk_size)

        except Exception as e:
            print(f"Skipping '{variable}' chunk {i}: {e}")
            continue

    return variable, stats


def get_stat_table(lres, hres, variables: Iterable[str]):
    """Function to write table with variable mean, min and max in lres and
    hres datasets (to check they match and their residuals are meaningful).
    """
    rows = [
        [
            var,
            lres.statistics["mean"][lres.name_to_index[var]],
            lres.statistics["minimum"][lres.name_to_index[var]],
            lres.statistics["maximum"][lres.name_to_index[var]],
            hres.statistics["mean"][hres.name_to_index[var]],
            hres.statistics["minimum"][hres.name_to_index[var]],
            hres.statistics["maximum"][hres.name_to_index[var]],
        ]
        for var in variables
    ]
    header = [
        "variable",
        "lres mean",
        "lres minimum",
        "lres maximum",
        "hres mean",
        "hres minimum",
        "hres maximum",
    ]
    align = ["<", ">", ">", ">", ">", ">", ">"]

    return table(rows, header, align)


def get_res_table(res_stats: dict):
    """Function to write table with residual mean, stdev, min and max."""
    rows = [
        [
            var,
            res_stats["mean"][var],
            res_stats["stdev"][var],
            res_stats["minimum"][var],
            res_stats["maximum"][var],
        ]
        for var in res_stats["mean"].keys()
    ]
    header = ["variable", "mean", "stdev", "minimum", "maximum"]
    align = ["<", ">", ">", ">", ">"]

    return table(rows, header, align)


def compute_residual_stats(args, selected_vars, hres_indices, lres_indices):
    # Load existing results if output file exists
    if os.path.exists(args.output):
        print(f"Loading existing statistics from {args.output}")
        existing_stats = np.load(args.output, allow_pickle=True).item()
        var_computed = set(existing_stats["mean"].keys())
    else:
        existing_stats = {"mean": {}, "stdev": {}, "minimum": {}, "maximum": {}}
        var_computed = set()

    # Select variables to compute
    if args.overwrite:
        var_to_compute = selected_vars
        print("Overwrite enabled: all selected variables will be recomputed.")
    else:
        var_to_compute = [v for v in selected_vars if v not in var_computed]

    if not var_to_compute:
        print("All requested variables already computed. Nothing to do.")
        print("\nResidual statistics summary:")
        print(get_res_table(existing_stats))
        return

    print(f"Computing stats for variables: {var_to_compute}")

    # Initialize per-variable stats
    running_stats = {
        var: {"mean": 0.0, "mean2": 0.0, "minimum": np.inf, "maximum": -np.inf}
        for var in var_to_compute
    }

    arguments = [(var, hres_indices, lres_indices, args) for var in var_to_compute]

    # Process data
    print(f"Launching {args.processes} worker(s)...")
    with Pool(processes=args.processes) as pool:
        imap = pool.imap_unordered(compute_variable_stats, arguments)

        for var, stats in tqdm.tqdm(imap, total=len(var_to_compute)):
            running_stats[var] = stats

    # Merge new stats into existing stats
    final_stats = existing_stats
    new_stats = finalize_stats(running_stats)
    for stat_name in ["mean", "stdev", "minimum", "maximum"]:
        final_stats[stat_name].update(new_stats[stat_name])

    print("\nResidual statistics summary:")
    print(get_res_table(final_stats))

    print(f"\nSaving statistics to: {args.output}")
    np.save(args.output, final_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Compute residual statistics between high-resolution "
        "and interpolated low-resolution Zarr fields.",
        usage="uv run residual-statistics hres.zarr lres.zarr interp_matrix.mat.npz "
        "[--output residual_stats.npy "
        "--start 2020-01-01T01:00:00 --end 2023-12-31T23:00:00 "
        "--chunk_size 10 --processes 3]",
    )
    parser.add_argument(
        "hres_zarr", type=str, help="Path to high-resolution Zarr dataset"
    )
    parser.add_argument(
        "lres_zarr", type=str, help="Path to low-resolution Zarr dataset"
    )
    parser.add_argument(
        "interp_matrix_npz",
        type=str,
        help="Path to sparse interpolation matrix in .npz format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="residual_stats.npy",
        help="Path to save .npy residual file",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Optional list of variables to compute stats for",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help=" Start date in the format YYYY-mm-ddTHH:MM:SS. "
        "Default is the first common date of the datasets.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help=" End date in the format YYYY-mm-ddTHH:MM:SS. "
        "Default is defined so that statistics are computed over 80% "
        "of the datasets duration (same as anemoi-datasets).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="Number of time steps per chunk. Default: 10.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Number of processes used to compute the statistics of the "
        "residuals for each variable in parallel. Default: 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing variables in the output file "
        "instead of skipping them.",
    )
    args = parser.parse_args()

    # Loading datasets
    hres_dataset = open_dataset(args.hres_zarr)
    lres_dataset = open_dataset(args.lres_zarr)

    if hres_dataset.frequency != lres_dataset.frequency:
        raise ValueError("High- and low-res datasets do not have the same frequency.")

    # Compute common start and end dates
    if args.start is None:
        args.start = max(hres_dataset.start_date, lres_dataset.start_date)

    if args.end is None:
        args.end = min(hres_dataset.end_date, lres_dataset.end_date)

        # Default: first 80% of the dataset
        dataset_end = min(hres_dataset.end_date, lres_dataset.end_date)
        total_duration = dataset_end - args.start
        args.end = args.start + total_duration * 8 // 10

    # Check for overlap
    if args.start >= args.end:
        raise ValueError("High- and low-res datasets do not have common dates.")

    # Build index lists
    hres_dates = set(
        hres_dataset.dates[
            np.logical_and(
                hres_dataset.dates >= np.datetime64(args.start),
                hres_dataset.dates < np.datetime64(args.end),
            )
        ]
    )

    lres_dates = set(
        lres_dataset.dates[
            np.logical_and(
                lres_dataset.dates >= np.datetime64(args.start),
                lres_dataset.dates < np.datetime64(args.end),
            )
        ]
    )

    hres_missing_dates = np.array([])
    if len(hres_dataset.missing):
        hres_missing_dates = hres_dataset.dates[hres_dataset.missing]

    lres_missing_dates = np.array([])
    if len(lres_dataset.missing):
        lres_missing_dates = lres_dataset.dates[lres_dataset.missing]

    hres_dates = hres_dates - set(hres_missing_dates)
    lres_dates = lres_dates - set(lres_missing_dates)

    common_dates = list(hres_dates.intersection(lres_dates))
    hres_indices = [
        hres_dataset.to_index(common_dates[i], hres_dataset.variables[0])[0]
        for i in range(len(common_dates))
    ]

    lres_indices = [
        lres_dataset.to_index(common_dates[i], lres_dataset.variables[0])[0]
        for i in range(len(common_dates))
    ]

    if len(hres_indices) != len(lres_indices):
        raise ValueError(
            "High- and low-res datasets do not have the same missing dates."
        )

    total_steps = len(hres_indices)

    print(
        f"Computing stats between {args.start} and {args.end} with "
        f"frequency {hres_dataset.frequency} ({total_steps} dates, "
        f"{len(hres_dataset.missing)} missing)."
    )

    shared_vars = set(hres_dataset.variables) & set(lres_dataset.variables)
    shared_vars = shared_vars - FORCING_VARIABLES

    if not shared_vars:
        raise ValueError("No common variables found between datasets.")

    if args.variables:
        requested = set(args.variables)
        available = shared_vars & requested
        missing = requested - shared_vars
        selected_vars = sorted(available)

        if missing:
            print(
                f"WARNING: some requested variables not found in both datasets: {sorted(missing)}"
            )
    else:
        selected_vars = shared_vars

    if not selected_vars:
        raise ValueError(
            f"None of the selected variables {args.variables} found in the datasets."
        )

    print("Selected variables:")
    print(get_stat_table(lres_dataset, hres_dataset, selected_vars))

    compute_residual_stats(args, selected_vars, hres_indices, lres_indices)


if __name__ == "__main__":
    main()
