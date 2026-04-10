#!/usr/bin/env python3
# language: python
"""
2026-04-10 MY with AI assistance

Filter FAOSTAT trade by fao_code (crosswalk), element code and year range.

RUN:
for export value:
python 01-read_FAOSTAT_tradedata.py --element 5922
OR for specific year range and export quantity (default element):
python 01-read_FAOSTAT_tradedata.py --start-year 1992 --end-year 2020


Usage examples

Default (interpolation ON):
python 01-read_FAOSTAT_tradedata.py

Interpolation OFF:
python 01-read_FAOSTAT_tradedata.py --no-timeseries-interpolation

You should now get:

trade_export_quantity_t_interpolated_1992_2020.csv (default)
trade_export_quantity_t_1992_2020.csv (when interpolation disabled)

"""

from pathlib import Path
import argparse

from typing import List, Any
import pandas as pd
import pickle


def _read_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter FAOSTAT trade by fao_code, element and year range.")
    p.add_argument(
        "--crosswalk",
        type=Path,
        default=Path("/Users/myliheik/Documents/myPython/FAOSTAT_trade/data/input/faostat_spam_crosswalk.csv"),
        help="Path to faostat_spam_crosswalk.csv"
    )
    p.add_argument(
        "--trade",
        type=Path,
        default=Path("/Users/myliheik/Documents/myPython/FAOSTAT_trade/data/input/Trade_CropsLivestock_E_All_Data_(Normalized)/Trade_CropsLivestock_E_All_Data_(Normalized).csv"),
        help="Path to FAOSTAT trade CSV"
    )
    p.add_argument(
        "--element",
        choices=["5910", "5922"],
        default="5910",
        help='Element code to filter: "5910" (export quantity) or "5922" (export value). Default: 5910'
    )
    p.add_argument(
        "--start-year",
        type=int,
        default=1992,
        help="Start year (inclusive). Default: 1992"
    )
    p.add_argument(
        "--end-year",
        type=int,
        default=2020,
        help="End year (inclusive). Default: 2020"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path. If omitted, a file is written to the parent[2] of the trade file with a generated name."
    )
    p.add_argument(
        "--meta-m49-isoa3",
        type=Path,
        default=Path("/Users/myliheik/Documents/myPython/FAOSTAT_trade/data/input/metaDictM49ISOa3.pkl"),
        help="Path to metaDictM49ISOa3.pkl"
    )
    p.add_argument(
        "--timeseries-interpolation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable time-series interpolation (default: enabled). Use --no-timeseries-interpolation to disable."
    )
    return p.parse_args(argv)


def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = list(df.columns)
    for cand in candidates:
        for c in cols:
            if c.strip().lower() == cand.lower():
                return c
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.strip().lower():
                return c
    return None


def read_trade(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, low_memory=False)


def filter_trade(
    trade: pd.DataFrame,
    fao_codes: List[str],
    element_code: str,
    start_year: int,
    end_year: int
) -> pd.DataFrame:
    item_col = find_col(trade, ["Item Code", "Item_Code", "ItemCode", "item code"])
    element_col = find_col(trade, ["Element Code", "Element_Code", "Element", "element"])
    year_col = find_col(trade, ["Year", "year", "Year_"])
    value_col = find_col(trade, ["Value", "value", "OBS_VALUE", "Value (kg)"])

    if not item_col or not element_col or not year_col:
        missing = [n for n, c in (("Item Code", item_col), ("Element", element_col), ("Year", year_col)) if not c]
        raise KeyError(f"Could not find required columns in trade data: {', '.join(missing)}")

    trade[item_col] = trade[item_col].astype(str).str.strip()
    trade[element_col] = trade[element_col].astype(str).str.strip()
    trade[year_col] = pd.to_numeric(trade[year_col], errors="coerce")

    mask_item = trade[item_col].isin(fao_codes)
    mask_element = trade[element_col] == str(element_code)
    mask_year = (trade[year_col] >= start_year) & (trade[year_col] <= end_year)

    filtered = trade[mask_item & mask_element & mask_year].copy()

    if value_col:
        filtered[value_col] = pd.to_numeric(filtered[value_col], errors="coerce")

    return filtered

def merge_with_crosswalk(filtered: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    item_col = find_col(filtered, ["Item Code", "Item_Code", "ItemCode", "item code"])
    if not item_col:
        raise KeyError("Could not find 'Item Code' column in filtered dataframe.")
    if "fao_code" not in crosswalk.columns:
        raise KeyError("Column 'fao_code' not found in crosswalk.")
    if "lu_class" not in crosswalk.columns:
        raise KeyError("Column 'lu_class' not found in crosswalk.")

    left = filtered.copy()
    right = crosswalk.copy()

    left[item_col] = left[item_col].astype(str).str.strip()
    right["fao_code"] = right["fao_code"].astype(str).str.strip()

    merged = left.merge(
        right[["fao_code", "lu_class"]].drop_duplicates(),
        left_on=item_col,
        right_on="fao_code",
        how="left",
    )
    return merged


def group_sum_by_lu_class(filtered: pd.DataFrame, sum_col: str | None = None) -> pd.DataFrame:
    """
    Group filtered data by 'lu_class' and return summed values per group.
    """
    if "lu_class" not in filtered.columns:
        raise KeyError("Column 'lu_class' not found in filtered dataframe.")

    tmp = filtered.copy()
    tmp[sum_col] = pd.to_numeric(tmp[sum_col], errors="coerce")

    grouped = (
        tmp.groupby(["Area Code (M49)", "Year", "lu_class"], dropna=False, as_index=False)[sum_col]
        .sum(min_count=1)
        .rename(columns={sum_col: "group_sum"})
    )
    return grouped


def add_isoa3_from_m49(grouped: pd.DataFrame, m49_to_isoa3: dict) -> pd.DataFrame:
    if "Area Code (M49)" not in grouped.columns:
        raise KeyError("Column 'Area Code (M49)' not found in grouped dataframe.")

    out = grouped.copy()
    # Normalize M49 Area Code to pandas nullable Int64
    # First remove . suffix codes (e.g. 002.03 for Africa)
    # The .03 suffix is FAO's convention for trade "mirror data" aggregates.
    # These are FAO aggregate/regional codes, not individual countries.

    # mask aggregate/regional codes containing "."
    m49_clean = (
        out["Area Code (M49)"]
        .str.lstrip("'")  # strip leading apostrophes
    )
    is_aggregate = m49_clean.str.contains(r"\.", regex=True, na=False)

    # convert to int only for non-aggregate rows
    m49_int = pd.to_numeric(m49_clean.where(~(is_aggregate)), errors="coerce").astype("Int64")

    out["ISOA3"] = m49_int.map(m49_to_isoa3)  # NaN for aggregates automatically
    return out

def drop_missing_isoa3(
    out: pd.DataFrame,
    m49_name_lookup: dict | None = None,
    n_examples: int = 10,
) -> tuple[pd.DataFrame, dict]:
    if "ISOA3" not in out.columns:
        raise KeyError("Column 'ISOA3' not found in dataframe.")

    excluded = out.loc[out["ISOA3"].isna()].copy()
    kept = out.loc[out["ISOA3"].notna()].copy()

    summary = {
        "rows_before": len(out),
        "rows_after": len(kept),
        "rows_dropped": len(excluded),
        "unique_m49_codes_dropped": (
            excluded["Area Code (M49)"].dropna().astype(str).nunique()
            if "Area Code (M49)" in excluded.columns
            else None
        ),
        "example_dropped_m49_codes": (
            excluded["Area Code (M49)"].dropna().astype(str).drop_duplicates().head(n_examples).tolist()
            if "Area Code (M49)" in excluded.columns
            else []
        ),
    }

    if m49_name_lookup is not None and "Area Code (M49)" in excluded.columns:
        m49_clean = (
            excluded["Area Code (M49)"]
            .str.lstrip("'")
        )
        is_aggregate = m49_clean.str.contains(r"\.", regex=True, na=False)
        m49_digits = m49_clean.where(~(is_aggregate)).str.extract(r"(\d+)")[0]
        m49_int = pd.to_numeric(m49_digits, errors="coerce").astype("Int64")

        excluded_names = (
            m49_int.map(m49_name_lookup)
            .dropna()
            .drop_duplicates()
            .head(n_examples)
            .tolist()
        )
        summary["example_dropped_country_names"] = excluded_names

    return kept, summary





def interpolate_group_timeseries(
    df: pd.DataFrame,
    value_col: str = "group_sum",
    group_cols: tuple[str, ...] = ("ISOA3", "lu_class"),
    year_col: str = "Year",
    start_year: int | None = None,
    end_year: int | None = None,
    interpolation_method: str = "linear",
    fill_both_directions: bool = True,
) -> pd.DataFrame:
    required = [*group_cols, year_col, value_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for interpolation: {missing}")

    out = df.copy()
    out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
    out[value_col] = pd.to_numeric(out[value_col], errors="coerce")

    if start_year is None:
        start_year = int(out[year_col].min())
    if end_year is None:
        end_year = int(out[year_col].max())

    full_time = pd.date_range(f"{start_year}-01-01", f"{end_year}-01-01", freq="YS")

    parts: list[pd.DataFrame] = []
    for keys, g in out.groupby(list(group_cols), dropna=False):
        g2 = g[[*group_cols, year_col, value_col]].copy()

        # Year -> datetime and set as ordered time index
        g2["Time"] = pd.to_datetime(g2[year_col].astype("string"), format="%Y", errors="coerce")
        g2 = g2.dropna(subset=["Time"]).set_index("Time").sort_index()

        # ensure full annual timeline exists
        g2 = g2.reindex(full_time)

        if not isinstance(keys, tuple):
            keys = (keys,)
        for c, k in zip(group_cols, keys):
            g2[c] = k

        if fill_both_directions:
            g2[value_col] = g2[value_col].interpolate(
                method=interpolation_method,
                limit_direction="both",
            )
        else:
            g2[value_col] = g2[value_col].interpolate(method=interpolation_method)

        g2[year_col] = g2.index.year.astype("Int64")
        g2 = g2.reset_index().rename(columns={"index": "Time"})
        parts.append(g2)

    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values([*group_cols, "Time"]).reset_index(drop=True)
    return result

def element_label_for_filename(element_code: str) -> str:
    labels = {
        "5910": "export_quantity_t",
        "5922": "export_value_1000_usd",
    }
    return labels.get(str(element_code), f"element_{element_code}")


def write_output(
    df: pd.DataFrame,
    out: Path | None,
    trade_path: Path,
    element: str,
    start: int,
    end: int,
    interpolated: bool,
) -> Path:
    if out is None:
        element_text = element_label_for_filename(element)
        data_dir = trade_path.parents[2]
        intermediate_dir = data_dir / "intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_interpolated" if interpolated else ""
        out = intermediate_dir / f"trade_{element_text}{suffix}_{start}_{end}.csv"

    df.to_csv(out, index=False)
    return out

######### Testing functions ########

def _test_add_isoa3_from_m49() -> None:
    lookup = {
        4: "AFG",
        20: "AND",
        203: "CZE",
        204: "BEN",
    }

    grouped = pd.DataFrame(
        {
            "Area Code (M49)": ["'204", "'203", "'020", "'004", "'002.03", None],
            "group_sum": [1, 2, 3, 4, 5, 6],
        }
    )

    result = add_isoa3_from_m49(grouped, lookup)

    # row order preserved
    assert result["Area Code (M49)"].tolist() == grouped["Area Code (M49)"].tolist()

    # regular country codes map correctly
    assert result["ISOA3"].tolist()[:4] == ["BEN", "CZE", "AND", "AFG"]

    # aggregate code with "." becomes missing
    assert pd.isna(result.loc[4, "ISOA3"])

    # missing input stays missing
    assert pd.isna(result.loc[5, "ISOA3"])

    # existing columns preserved
    assert result["group_sum"].tolist() == [1, 2, 3, 4, 5, 6]

######### MAIN #########

def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.crosswalk.exists():
        raise FileNotFoundError(f"Crosswalk file not found: {args.crosswalk}")
    if not args.trade.exists():
        raise FileNotFoundError(f"Trade file not found: {args.trade}")

    crosswalk_df = pd.read_csv(args.crosswalk, dtype=str)
    # Check:
    #print(len(crosswalk_df), len(crosswalk_df[["fao_code", "lu_class"]].drop_duplicates()))
    #print(len(crosswalk_df["fao_code"].astype(str).str.strip().dropna().unique().tolist()))
    #print(crosswalk_df.head(10))

    fao_codes = (
        crosswalk_df["fao_code"].astype(str).str.strip().dropna().unique().tolist()
    )

    trade_df = read_trade(args.trade)
    filtered = filter_trade(trade_df, fao_codes, args.element, args.start_year, args.end_year)
    
    merged = merge_with_crosswalk(filtered, crosswalk_df)
    print(merged.head(10))
    grouped = group_sum_by_lu_class(merged, "Value")
    
    if not args.meta_m49_isoa3.exists():
        raise FileNotFoundError(f"metaDictM49ISOa3 file not found: {args.meta_m49_isoa3}")

    metaDictM49ISOa3 = _read_pickle(args.meta_m49_isoa3)

    grouped = group_sum_by_lu_class(merged, "Value")
    grouped = add_isoa3_from_m49(grouped, metaDictM49ISOa3)
    filtered, drop_summary = drop_missing_isoa3(grouped, m49_name_lookup=metaDictM49ISOa3, n_examples=10)

    if args.timeseries_interpolation:
        output_df = interpolate_group_timeseries(
            filtered,
            value_col="group_sum",
            group_cols=("ISOA3", "lu_class"),
            year_col="Year",
            start_year=args.start_year,
            end_year=args.end_year,
            interpolation_method="linear",
            fill_both_directions=True,
        )
    else:
        output_df = filtered

    print(drop_summary)
    print(grouped.head(10))
    print(filtered.head(10))

    out_path = write_output(
        output_df,
        args.out,
        args.trade,
        args.element,
        args.start_year,
        args.end_year,
        interpolated=args.timeseries_interpolation,
    )
    print(f"Data written to: {out_path}")


    print("Testing...")
    _test_add_isoa3_from_m49()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

