#!/usr/bin/env python3
"""
2026-04-10 MY with AI assistance

RUN:

python 02_faostat_to_tif.py \
  --input-file "/Users/myliheik/Documents/myPython/FAOSTAT_trade/data/intermediate/trade_export_quantity_t_1992_2020.csv" \
  --start-year 1992 \
  --end-year 2020

python 02_faostat_to_tif.py \
  --input-file  /Users/myliheik/Documents/myPython/FAOSTAT_trade/data/intermediate/trade_export_value_1000_usd_1992_2020.csv \
  --start-year 1992 \
  --end-year 2020 


Create annual FAOSTAT export rasters by:
1) filtering data by user-defined year range,
2) aggregating country-level annual values,
3) merging with Natural Earth ADM0 polygons, and
4) rasterizing each year to GeoTIFF.

User options
------------
--input-file PATH
    Required. FAOSTAT CSV file to process.

--start-year INT
    Required. First year to include (inclusive).

--end-year INT
    Required. Last year to include (inclusive).

--ne-file PATH
    Optional. Path to Natural Earth shapefile.
    Default: /Users/myliheik/Documents/myPython/FAOSTAT_trade/data/gis/adm0_NatEarth_all_ids.shp

--faostat-key COLUMN
    Optional. Country key column in FAOSTAT CSV.
    Default: "ISOA3"

--boundary-key COLUMN
    Optional. Country key column in the Natural Earth shapefile.
    Default: "iso_a3"

--year-col COLUMN
    Optional. Year column name in FAOSTAT CSV.
    Default: "Year"

--value-col COLUMN
    Optional. Numeric value column to rasterize.
    Default: "group_sum"

--resolution FLOAT
    Optional. Output raster resolution in CRS units.
    Default: 0.08333333333333333 (5 arc-minutes for EPSG:4326)


Output
------
GeoTIFF files are written to:
    ../output
relative to the input file directory.
The output directory is created automatically if it does not exist.
 

"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

NODATA_VALUE = -9999.0

def ensure_output_dir(input_file: Path) -> Path:
    output_dir = (input_file.parent / ".." / "output").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def validate_years(start_year: int, end_year: int) -> None:
    if start_year > end_year:
        raise ValueError(f"start-year ({start_year}) must be <= end-year ({end_year}).")


def load_faostat(path: Path, year_col: str, value_col: str, key_col: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    required = [year_col, value_col, key_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required FAOSTAT columns: {missing}")

    df = df.copy()
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df[key_col] = df[key_col].astype(str).str.strip()

    return df



def load_boundaries(path: Path, key_col: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if key_col not in gdf.columns:
        raise KeyError(
            f"Boundary key column not found: {key_col}. "
            f"Available columns: {list(gdf.columns)}"
        )
    gdf = gdf.copy()
    gdf[key_col] = gdf[key_col].astype(str).str.strip()
    return gdf


def aggregate_year(df: pd.DataFrame, year: int, year_col: str, key_col: str, value_col: str) -> pd.DataFrame:
    year_df = df.loc[df[year_col] == year, [key_col, value_col]].dropna(subset=[key_col])
    agg = (
        year_df.groupby(key_col, as_index=False)[value_col]
        .sum(min_count=1)
    )
    return agg



def merge_to_boundaries(
    boundaries: gpd.GeoDataFrame,
    agg: pd.DataFrame,
    boundary_key: str,
    faostat_key: str,
    value_col: str,
    nodata_value: float = -9999.0,
) -> gpd.GeoDataFrame:
    merged = boundaries.merge(
        agg,
        how="left",
        left_on=boundary_key,
        right_on=faostat_key,
    )
    merged[value_col] = pd.to_numeric(merged[value_col], errors="coerce").fillna(nodata_value).astype("float32")
    return merged


def raster_meta_from_gdf(gdf: gpd.GeoDataFrame, resolution: float) -> tuple[rasterio.Affine, int, int]:
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = from_origin(minx, maxy, resolution, resolution)
    return transform, width, height


def rasterize_year(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    transform: rasterio.Affine,
    width: int,
    height: int,
) -> np.ndarray:
    shapes: Iterable[tuple[object, float]] = (
        (geom, float(val))
        for geom, val in zip(gdf.geometry, gdf[value_col])
        if geom is not None and not np.isnan(val)
    )
    arr = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        transform=transform,
        fill=NODATA_VALUE,      # <- was np.nan
        dtype="float32",
        all_touched=True,
    )
    return arr.astype("float32")



def save_raster_cube(
    band_arrays: list[np.ndarray],
    years: list[int],
    out_file: Path,
    crs,
    transform: rasterio.Affine,
    nodata_value: float = NODATA_VALUE,
) -> Path:
    if len(band_arrays) != len(years):
        raise ValueError("band_arrays and years must have same length.")
    if not band_arrays:
        raise ValueError("band_arrays is empty.")

    height, width = band_arrays[0].shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(band_arrays),
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata_value,
        "compress": "lzw",
    }

    with rasterio.open(out_file, "w", **profile) as dst:
        for i, (arr, yr) in enumerate(zip(band_arrays, years), start=1):
            dst.write(arr.astype("float32"), i)
            dst.set_band_description(i, str(yr))
        dst.update_tags(years=",".join(map(str, years)))

    return out_file

def summarize_band_arrays(band_arrays: list[np.ndarray], years: list[int]) -> pd.DataFrame:
    if len(band_arrays) != len(years):
        raise ValueError("band_arrays and years must have same length.")
    if not band_arrays:
        return pd.DataFrame(columns=["band", "year", "shape", "valid_cells", "na_cells", "min", "max", "mean", "sum"])

    rows = []
    for i, (arr, yr) in enumerate(zip(band_arrays, years), start=1):
        arr = np.asarray(arr, dtype="float64")
        valid_mask = ~np.isnan(arr)
        valid_n = int(valid_mask.sum())
        na_n = int((~valid_mask).sum())

        rows.append(
            {
                "band": i,
                "year": int(yr),
                "shape": f"{arr.shape[0]}x{arr.shape[1]}",
                "valid_cells": valid_n,
                "na_cells": na_n,
                "min": float(np.nanmin(arr)) if valid_n else np.nan,
                "max": float(np.nanmax(arr)) if valid_n else np.nan,
                "mean": float(np.nanmean(arr)) if valid_n else np.nan,
                "sum": float(np.nansum(arr)) if valid_n else np.nan,
            }
        )

    return pd.DataFrame(rows)

###################### MAIN FUNCTION ######################

def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
   

    validate_years(args.start_year, args.end_year)
    output_dir = ensure_output_dir(args.input_file)

    faostat = load_faostat(args.input_file, args.year_col, args.value_col, args.faostat_key)
    print(faostat.head(10))
    
    boundaries = load_boundaries(args.ne_file, args.boundary_key)

    transform, width, height = raster_meta_from_gdf(boundaries, args.resolution)

    years = list(range(args.start_year, args.end_year + 1))  
    stem = "_".join(args.input_file.stem.split("_")[:-2])

    print(stem)
    print(f"Processing years {years} for FAOSTAT data from {args.input_file}...")

    for lu_class in faostat['lu_class'].unique():
        fao_lu = faostat.loc[faostat["lu_class"] == lu_class].copy()

        band_arrays: list[np.ndarray] = []

        for year in years:
            agg = aggregate_year(
                df=fao_lu,
                year=year,
                year_col=args.year_col,
                key_col=args.faostat_key,
                value_col=args.value_col,
            )
            merged = merge_to_boundaries(
                boundaries=boundaries,
                agg=agg,
                boundary_key=args.boundary_key,
                faostat_key=args.faostat_key,
                value_col=args.value_col,
                nodata_value=NODATA_VALUE,
            )
            arr = rasterize_year(
                gdf=merged,
                value_col=args.value_col,
                transform=transform,
                width=width,
                height=height,
            )

            band_arrays.append(arr)

        if args.start_year == args.end_year:
            year_suffix = str(args.start_year)
        else:            
            year_suffix = f"{args.start_year}_{args.end_year}"

        out_tif = output_dir / f"{stem}_{lu_class}_{year_suffix}_cube.tif"
        saved_path = save_raster_cube(
            band_arrays=band_arrays,
            years=years,
            out_file=out_tif,
            crs=boundaries.crs,
            transform=transform,
            nodata_value=NODATA_VALUE,
        )
        logging.info("Saved cube %s", out_tif)
        #print(f'Saved cube {saved_path}')
        # If you want to check band summaries:
        #summary_df = summarize_band_arrays(band_arrays, years)
        #logging.info("Band summary for lu_class=%s\n%s", lu_class, summary_df.to_string(index=False))

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create annual FAOSTAT export rasters merged with Natural Earth ADM0 boundaries."
    )
    parser.add_argument("--input-file", type=Path, required=True, help="Path to FAOSTAT CSV input file.")
    parser.add_argument("--start-year", type=int, required=True, help="Start year (inclusive).")
    parser.add_argument("--end-year", type=int, required=True, help="End year (inclusive).")

    parser.add_argument(
        "--ne-file",
        type=Path,
        default=Path("/Users/myliheik/Documents/myPython/FAOSTAT_trade/data/gis/adm0_NatEarth_all_ids.shp"),
        help="Path to Natural Earth shapefile.",
    )
    parser.add_argument(
        "--faostat-key",
        default="ISOA3",
        help="FAOSTAT country key column (e.g. 'Area Code (ISO3)').",
    )
    parser.add_argument(
        "--boundary-key",
        default="iso_a3",
        help="Country key column in the Natural Earth shapefile.",
    )
    parser.add_argument(
        "--year-col",
        default="Year",
        help="FAOSTAT year column name.",
    )
    parser.add_argument(
        "--value-col",
        default="group_sum",
        help="FAOSTAT numeric value column to rasterize.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=5 / 60,
        help="Raster resolution in CRS units. For EPSG:4326, 5 arc-minutes = 5/60 degrees.",
    )
    args = parser.parse_args()

    raise SystemExit(main(args))

