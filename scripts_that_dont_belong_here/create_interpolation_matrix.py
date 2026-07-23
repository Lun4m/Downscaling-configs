"""This script computes an interpolation matrix between two grids, and stores
it in a .npz file.

Scipy's nearest neighbour or linear interpolation method can be used.
In the case of the nearest neighbour interpolation, the number of neighbour
points in the source grid, and the power involved in the weighting can be
chosen by the user.

!!! Important !!!
- All the longitudes / latitudes should be in degrees.
- The longitude convention (-180~180 or 0~360) should be the same for all grids.

Example:
  uv run get-interpolation-matrix \\
    --source lres.zarr \\
    --target hres.zarr \\
    --output lres_to_hres.npz \\
    --method nearest \\
    --n_neighbours 3 \\
    --power 2
"""

import os
import warnings
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse
import scipy.spatial
import xarray as xr


@dataclass
class CliArgs:
    source: str
    target: str
    output: str
    method: Literal["linear", "nearest"]
    n_neighbours: int
    power: float
    overwrite: bool

    @classmethod
    def parse(cls):
        ap = ArgumentParser(
            description=__doc__, formatter_class=RawDescriptionHelpFormatter
        )
        ap.add_argument(
            "--source",
            type=str,
            required=True,
            help="Source grid file name (zarr)",
        )
        ap.add_argument(
            "--target",
            type=str,
            required=True,
            help="Target grid file name (zarr)",
        )
        ap.add_argument(
            "--output",
            type=str,
            required=True,
            help="Name of the output matrix file (npz)",
        )
        ap.add_argument(
            "--method",
            type=str,
            required=True,
            choices=["linear", "nearest"],
            help="Interpolation method used. Either 'linear' or 'nearest'.",
        )
        ap.add_argument(
            "--n_neighbours",
            type=int,
            default=1,
            help="Number of nearest neighbours (default is 1).",
        )
        ap.add_argument(
            "--power",
            type=float,
            default=1.0,
            help="The weights of the interpolation are proportional to 1 / (d ** power), "
            "where d is the distance of target grid points to source grid points.",
        )
        ap.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite interpolation matrix file ",
        )

        args = ap.parse_args()
        return cls(**vars(args))


def read_grid_zarr(path: str) -> np.ndarray:
    with xr.open_zarr(path, consolidated=False) as ds:
        lats = ds["latitudes"].values
        lons = ds["longitudes"].values

    return np.stack((lons, lats), axis=-1)


def check_target_in_source_domain(source: np.ndarray, target: np.ndarray):
    """Check that target domain is inside source domain.
    This is useful in the case of 'linear' interpolation method
    (there is a risk that the script fails at computing the interpolation
    matrix if the target domain is bigger than the source domain)
    """

    source_lons, source_lats = np.split(source, 2, axis=-1)
    target_lons, target_lats = np.split(target, 2, axis=-1)

    if target_lons.min() < source_lons.min():
        warnings.warn(
            f"\nWarning: Extrapolation detected. "
            f"The western boundary of the target grid "
            f"({target_lons.min()}) is further west than that of the "
            f"source grid ({source_lons.min()})."
            f"This might cause a problem if the method used is 'linear'",
            stacklevel=2,
        )
    if target_lons.max() > source_lons.max():
        warnings.warn(
            f"\nWarning: Extrapolation detected. "
            f"The eastern boundary of the target grid "
            f"({target_lons.max()}) is further east than that of the "
            f"source grid ({source_lons.max()})."
            f"This might cause a problem if the method used is 'linear'",
            stacklevel=2,
        )
    if target_lats.min() < source_lats.min():
        warnings.warn(
            f"\nWarning: Extrapolation detected. "
            f"The southern boundary of the target grid "
            f"({target_lats.min()}) is further south than that of the "
            f"source grid ({source_lats.min()})."
            f"This might cause a problem if the method used is 'linear'",
            stacklevel=2,
        )
    if target_lats.max() > source_lats.max():
        warnings.warn(
            f"\nWarning: Extrapolation detected. "
            f"The northern boundary of the target grid "
            f"({target_lats.max()}) is further north than that of the "
            f"source grid ({source_lats.max()})."
            f"This might cause a problem if the method used is 'linear'",
            stacklevel=2,
        )


def build_sparse_interpolation_matrix(
    indices: np.ndarray, weights: np.ndarray, N_in: int
) -> scipy.sparse.csr_matrix:
    """Build a sparse interpolation matrix.

    Parameters:
    - indices: shape (target, N_neighbours)
    - weights: shape (target, N_neighbours)
    - N_in: total number of source points

    Returns:
    - shape (target, source)
    """
    N_out, N_neighbours = indices.shape
    rows = np.repeat(np.arange(N_out), N_neighbours)
    cols = indices.flatten()
    data = weights.flatten()

    # Keep only valid column indices
    valid_mask = (cols >= 0) & np.isfinite(data) & (data != 0.0)

    rows = rows[valid_mask]
    cols = cols[valid_mask]
    data = data[valid_mask]

    matrix = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(N_out, N_in))
    return matrix


def compute_nearest_neighbour_weights(
    points_in: np.ndarray, points_out: np.ndarray, n_neighbours: int, power: float
) -> tuple[np.ndarray, np.ndarray]:
    # Tolerance parameter
    epsilon = 1.0e-12

    # Get index of source grid closest to every target point
    tree = scipy.spatial.KDTree(points_in)
    distances, indices = tree.query(points_out, k=n_neighbours)
    weights = 1.0 / (distances**power + epsilon)

    # Add degenerate dimension in case n_neighbours = 1
    if indices.ndim == 1:
        indices, weights = indices[:, None], weights[:, None]

    # Normalizing of weights
    weights /= weights.sum(axis=1)[:, None]
    return indices, weights


def compute_linear_weights(
    points_in: np.ndarray, points_out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Look for points in source grid that are closest to the target points
    tri = scipy.spatial.Delaunay(points_in)
    simplex = tri.find_simplex(points_out)

    # Manage points outside the domain : simplex = -1
    d = points_in.shape[1]
    indices = np.full((len(points_out), d + 1), -1, dtype=int)
    weights = np.zeros((len(points_out), d + 1))

    mask = simplex >= 0
    if np.any(mask):
        x = tri.transform[simplex[mask], :d]
        y = points_out[mask] - tri.transform[simplex[mask], d]
        bary = np.einsum("ijk,ik->ij", x, y)
        bary_coords = np.c_[bary, 1 - bary.sum(axis=1)]
        vertices = tri.simplices[simplex[mask]]
        indices[mask, :] = vertices
        weights[mask, :] = bary_coords
    return indices, weights


def save_interpolation_matrix(args: CliArgs):
    if os.path.exists(args.output) and not args.overwrite:
        print(f"Error. Interpolation matrix file {args.output} already exists.")
        raise SystemExit

    print(f" => Creating {args.output}")

    source = read_grid_zarr(args.source)
    target = read_grid_zarr(args.target)

    print(f"{source=}")
    print(f"{target=}")

    # Check if target domain is likely to be in source domain
    check_target_in_source_domain(source, target)

    match args.method:
        case "nearest":
            indices, weights = compute_nearest_neighbour_weights(
                source, target, args.n_neighbours, args.power
            )

        case "linear":
            indices, weights = compute_linear_weights(source, target)

    interpolation_matrix = build_sparse_interpolation_matrix(
        indices, weights, np.shape(source)[0]
    )

    scipy.sparse.save_npz(args.output, interpolation_matrix)


def main():
    args = CliArgs.parse()
    save_interpolation_matrix(args)


if __name__ == "__main__":
    main()
