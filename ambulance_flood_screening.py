from pathlib import Path
import math

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from shapely import wkt
from shapely.geometry import LineString
from tqdm import tqdm


# Configuration

CALLS_XLSX = Path("sos_alarm_amb.xlsx")
FLOOD_NC = Path("Data/Flood/Fidx.nc")
MUNICIPALITIES = Path("Data/Municipalities/alla_kommuner.shp")
GRAPH_PATH = Path("Data/Roads/sweden_road_network_all_components.graphml")

ROUTING_DIR = Path("rerouted_all_components")
OUTPUT_DIR = Path("final_analysis_all_components")

ROUTES_GPKG = ROUTING_DIR / "ambulance_routes_all_components.gpkg"
ROUTING_FAILURES = ROUTING_DIR / "routing_failures.csv"

HISTORICAL_CSV = OUTPUT_DIR / "historical_exposure_2015_2021.csv"
HISTORICAL_GPKG = OUTPUT_DIR / "routes_historical_exposure_2015_2021.gpkg"
HISTORICAL_YEARLY_CSV = OUTPUT_DIR / "historical_exposure_yearly_summary_2015_2021.csv"

PHASE2_ROUTES_CSV = OUTPUT_DIR / "final_phase2_2021_routes.csv"
CANDIDATE_PAIRS_CSV = OUTPUT_DIR / "final_phase2_candidate_pairs_median.csv"
MUNICIPALITY_STATS_CSV = OUTPUT_DIR / "final_phase2_municipality_stats.csv"
LENGTH_QUARTILES_CSV = OUTPUT_DIR / "final_phase2_length_quartiles.csv"
CANDIDATE_ROUTES_GPKG = OUTPUT_DIR / "final_phase2_candidate_routes.gpkg"
SUMMARY_TXT = OUTPUT_DIR / "final_phase2_summary.txt"

CRS_WGS84 = "EPSG:4326"
CRS_ANALYSIS = "EPSG:3006"

DFI_VARIABLE = "flood_index"
DFI_THRESHOLD = 0.0
YEAR_START = 2015
YEAR_END = 2021
OPERATIONAL_YEAR = 2021
SAMPLE_DISTANCE_M = 100.0

REBUILD_GRAPH = False
RERUN_ROUTING = True


# Data loading

def safe_wkt(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value == "POINT()":
        return None
    try:
        return wkt.loads(value)
    except Exception:
        return None


def load_calls():
    calls = pd.read_excel(CALLS_XLSX)
    calls.columns = calls.columns.str.strip()

    calls["geometry"] = calls["position"].apply(safe_wkt)
    calls["amb_geometry"] = calls["amb_position"].apply(safe_wkt)
    calls = calls[calls["geometry"].notna()].copy()
    calls = calls.reset_index(drop=True)
    calls["call_uid"] = calls.index + 1

    calls["status_T"] = pd.to_datetime(calls["status_T"], errors="coerce")
    calls["status_F"] = pd.to_datetime(calls["status_F"], errors="coerce")
    calls["response_time_sec"] = (
        calls["status_F"] - calls["status_T"]
    ).dt.total_seconds()

    patient = gpd.GeoSeries(
        calls["geometry"], crs=CRS_WGS84
    ).to_crs(CRS_ANALYSIS)

    ambulance = gpd.GeoSeries(
        calls["amb_geometry"], crs=CRS_WGS84
    ).to_crs(CRS_ANALYSIS)

    calls["patient_geometry_utm"] = patient.values
    calls["amb_geometry_utm"] = ambulance.values

    return calls


def load_dfi():
    ds = xr.open_dataset(FLOOD_NC)
    da = ds[DFI_VARIABLE]

    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

    da = da.sel(
        time=slice(
            f"{YEAR_START}-01-01",
            f"{YEAR_END}-12-31",
        )
    )
    da = da.transpose("time", "lat", "lon")

    dates = pd.DatetimeIndex(
        pd.to_datetime(da["time"].values)
    ).normalize()

    lat = da["lat"].values
    lon = da["lon"].values
    values = da.values

    date_lookup = {
        date: i for i, date in enumerate(dates)
    }

    return dates, lat, lon, values, date_lookup


# Road network and routing

def load_or_build_graph():
    if GRAPH_PATH.exists() and not REBUILD_GRAPH:
        graph = ox.load_graphml(GRAPH_PATH)
    else:
        graph = ox.graph_from_place(
            "Sweden",
            network_type="drive",
            simplify=True,
            retain_all=True,
        )
        graph = ox.project_graph(
            graph,
            to_crs=CRS_ANALYSIS,
        )
        GRAPH_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        ox.save_graphml(
            graph,
            GRAPH_PATH,
        )

    nodes, _ = ox.graph_to_gdfs(
        graph,
        nodes=True,
        edges=True,
    )

    if nodes.crs is None or nodes.crs.to_string() != CRS_ANALYSIS:
        graph = ox.project_graph(
            graph,
            to_crs=CRS_ANALYSIS,
        )

    return graph


def route_line_from_nodes(graph, route_nodes):
    coords = [
        (
            graph.nodes[n]["x"],
            graph.nodes[n]["y"],
        )
        for n in route_nodes
    ]

    clean = []
    for coord in coords:
        if not clean or coord != clean[-1]:
            clean.append(coord)

    if len(clean) < 2:
        return None

    return LineString(clean)


def route_calls(calls, graph):
    eligible = calls[
        calls["amb_geometry_utm"].notna()
    ].copy()

    start_x = np.array([
        g.x for g in eligible["amb_geometry_utm"]
    ])
    start_y = np.array([
        g.y for g in eligible["amb_geometry_utm"]
    ])
    end_x = np.array([
        g.x for g in eligible["patient_geometry_utm"]
    ])
    end_y = np.array([
        g.y for g in eligible["patient_geometry_utm"]
    ])

    start_nodes, start_dist = ox.distance.nearest_nodes(
        graph,
        X=start_x,
        Y=start_y,
        return_dist=True,
    )

    end_nodes, end_dist = ox.distance.nearest_nodes(
        graph,
        X=end_x,
        Y=end_y,
        return_dist=True,
    )

    records = []
    failures = []

    iterator = zip(
        eligible.itertuples(index=False),
        start_nodes,
        end_nodes,
        start_dist,
        end_dist,
    )

    for row, start_node, end_node, start_snap, end_snap in tqdm(
        iterator,
        total=len(eligible),
        desc="Routing",
    ):
        try:
            route_nodes = nx.shortest_path(
                graph,
                start_node,
                end_node,
                weight="length",
            )
        except nx.NetworkXNoPath:
            failures.append({
                "call_id": row.call_uid,
                "reason": "no_path",
            })
            continue
        except Exception:
            failures.append({
                "call_id": row.call_uid,
                "reason": "routing_error",
            })
            continue

        route_line = route_line_from_nodes(
            graph,
            route_nodes,
        )

        if route_line is None or route_line.is_empty:
            failures.append({
                "call_id": row.call_uid,
                "reason": "empty_route",
            })
            continue

        route_length_m = nx.path_weight(
            graph,
            route_nodes,
            weight="length",
        )

        records.append({
            "call_id": row.call_uid,
            "status_T": row.status_T,
            "response_time_sec": row.response_time_sec,
            "route_length_m": float(route_length_m),
            "start_snap_m": float(start_snap),
            "end_snap_m": float(end_snap),
            "geometry": route_line,
        })

    routes = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=CRS_ANALYSIS,
    )

    failures = pd.DataFrame(failures)

    ROUTING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    routes.to_file(
        ROUTES_GPKG,
        driver="GPKG",
    )

    failures.to_csv(
        ROUTING_FAILURES,
        index=False,
    )

    print(f"Eligible calls: {len(eligible):,}")
    print(f"Successful routes: {len(routes):,}")
    print(f"Routing failures: {len(failures):,}")

    return routes


# Route sampling and DFI lookup

def densify_line(
    line,
    spacing_m=SAMPLE_DISTANCE_M,
):
    if line is None or line.is_empty:
        return np.empty((0, 2))

    length = line.length
    if length <= 0:
        return np.empty((0, 2))

    n_steps = max(
        1,
        int(math.ceil(length / spacing_m)),
    )

    distances = np.linspace(
        0,
        length,
        n_steps + 1,
    )

    points = [
        line.interpolate(distance)
        for distance in distances
    ]

    return np.array([
        (point.x, point.y)
        for point in points
    ])


def nearest_grid_indices(
    coords_utm,
    transformer,
    lat_values,
    lon_values,
):
    if len(coords_utm) == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=int),
        )

    lon, lat = transformer.transform(
        coords_utm[:, 0],
        coords_utm[:, 1],
    )

    lat_idx = np.array(
        [
            int(np.abs(lat_values - value).argmin())
            for value in lat
        ],
        dtype=int,
    )

    lon_idx = np.array(
        [
            int(np.abs(lon_values - value).argmin())
            for value in lon
        ],
        dtype=int,
    )

    return lat_idx, lon_idx


# Historical exposure

def historical_exposure(
    routes,
    dates,
    lat_values,
    lon_values,
    dfi_values,
):
    transformer = Transformer.from_crs(
        CRS_ANALYSIS,
        CRS_WGS84,
        always_xy=True,
    )

    years = np.arange(
        YEAR_START,
        YEAR_END + 1,
    )

    year_masks = {
        year: dates.year == year
        for year in years
    }

    records = []

    for row in tqdm(
        routes.itertuples(index=False),
        total=len(routes),
        desc="Historical exposure",
    ):
        coords = densify_line(
            row.geometry
        )

        lat_idx, lon_idx = nearest_grid_indices(
            coords,
            transformer,
            lat_values,
            lon_values,
        )

        cells = np.unique(
            np.column_stack([
                lat_idx,
                lon_idx,
            ]),
            axis=0,
        )

        route_dfi = dfi_values[
            :,
            cells[:, 0],
            cells[:, 1],
        ]

        if route_dfi.ndim == 1:
            route_dfi = route_dfi[:, None]

        valid_days = np.any(
            np.isfinite(route_dfi),
            axis=1,
        )

        exposed_days = (
            np.any(
                route_dfi > DFI_THRESHOLD,
                axis=1,
            )
            & valid_days
        )

        record = {
            "call_id": row.call_id,
            "n_valid_days": int(valid_days.sum()),
            "n_exposed_days": int(exposed_days.sum()),
            "annualised_exposed_days": float(
                exposed_days.sum() / len(years)
            ),
            "n_unique_dfi_cells": int(len(cells)),
        }

        for year in years:
            record[f"exposed_days_{year}"] = int(
                exposed_days[
                    year_masks[year]
                ].sum()
            )

        records.append(record)

    exposure = pd.DataFrame(records)

    result = routes.merge(
        exposure,
        on="call_id",
        how="left",
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    exposure.to_csv(
        HISTORICAL_CSV,
        index=False,
    )

    result.to_file(
        HISTORICAL_GPKG,
        driver="GPKG",
    )

    yearly = pd.DataFrame({
        "year": years,
        "mean_exposed_days": [
            exposure[
                f"exposed_days_{year}"
            ].mean()
            for year in years
        ],
        "median_exposed_days": [
            exposure[
                f"exposed_days_{year}"
            ].median()
            for year in years
        ],
    })

    yearly.to_csv(
        HISTORICAL_YEARLY_CSV,
        index=False,
    )

    print(
        "Historical exposure: "
        f"median={exposure['n_exposed_days'].median():.0f}, "
        f"max={exposure['n_exposed_days'].max():.0f}"
    )

    return gpd.GeoDataFrame(
        result,
        geometry="geometry",
        crs=routes.crs,
    )


# Operational screening

def same_day_exposure(
    row,
    date_lookup,
    transformer,
    lat_values,
    lon_values,
    dfi_values,
):
    date = pd.Timestamp(
        row.status_T
    ).normalize()

    time_idx = date_lookup.get(date)

    if time_idx is None:
        return False, np.nan

    coords = densify_line(
        row.geometry
    )

    if len(coords) == 0:
        return False, np.nan

    lon, lat = transformer.transform(
        coords[:, 0],
        coords[:, 1],
    )

    flood_slice = dfi_values[
        time_idx,
        :,
        :,
    ]

    interpolator = RegularGridInterpolator(
        (lat_values, lon_values),
        flood_slice,
        bounds_error=False,
        fill_value=0.0,
    )

    points = np.column_stack([
        lat,
        lon,
    ])

    values = interpolator(points)
    valid = np.isfinite(values)

    if not valid.any():
        return False, np.nan

    exposed = (
        values[valid] > DFI_THRESHOLD
    )

    return (
        bool(exposed.any()),
        float(exposed.mean()),
    )


def run_phase2(
    routes,
    date_lookup,
    lat_values,
    lon_values,
    dfi_values,
):
    phase2 = routes[
        routes["status_T"].dt.year.eq(
            OPERATIONAL_YEAR
        )
        & routes["response_time_sec"].notna()
        & routes["response_time_sec"].gt(0)
        & routes["route_length_m"].gt(0)
    ].copy()

    phase2["route_length_km"] = (
        phase2["route_length_m"] / 1000.0
    )

    phase2[
        "distance_normalised_rt_s_per_km"
    ] = (
        phase2["response_time_sec"]
        / phase2["route_length_km"]
    )

    transformer = Transformer.from_crs(
        CRS_ANALYSIS,
        CRS_WGS84,
        always_xy=True,
    )

    exposed_flag = []
    exposure_fraction = []

    for row in tqdm(
        phase2.itertuples(index=False),
        total=len(phase2),
        desc="2021 exposure",
    ):
        flag, fraction = same_day_exposure(
            row,
            date_lookup,
            transformer,
            lat_values,
            lon_values,
            dfi_values,
        )

        exposed_flag.append(flag)
        exposure_fraction.append(fraction)

    phase2["exposed_flag"] = exposed_flag
    phase2["exposure_fraction"] = exposure_fraction

    phase2["length_quartile"] = pd.qcut(
        phase2["route_length_km"],
        4,
        labels=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
        ],
    )

    municipalities = gpd.read_file(
        MUNICIPALITIES
    ).to_crs(
        CRS_ANALYSIS
    )

    pairs = gpd.sjoin(
        phase2,
        municipalities[
            ["KOM_NAMN", "geometry"]
        ],
        how="left",
        predicate="intersects",
    ).drop(
        columns=["index_right"],
        errors="ignore",
    )

    nonexposed = pairs[
        ~pairs["exposed_flag"]
    ].copy()

    baselines = (
        nonexposed
        .groupby("KOM_NAMN")[
            "distance_normalised_rt_s_per_km"
        ]
        .agg(
            median_nonexposed_s_per_km="median",
            p75_nonexposed_s_per_km=lambda x: x.quantile(0.75),
            n_nonexposed="size",
        )
        .reset_index()
    )

    exposed_counts = (
        pairs[
            pairs["exposed_flag"]
        ]
        .groupby("KOM_NAMN")[
            "call_id"
        ]
        .nunique()
        .rename("n_exposed")
        .reset_index()
    )

    municipality_stats = baselines.merge(
        exposed_counts,
        on="KOM_NAMN",
        how="outer",
    )

    pairs = pairs.merge(
        municipality_stats,
        on="KOM_NAMN",
        how="left",
    )

    candidate_pairs = pairs[
        pairs["exposed_flag"]
        & pairs[
            "median_nonexposed_s_per_km"
        ].notna()
        & (
            pairs[
                "distance_normalised_rt_s_per_km"
            ]
            > pairs[
                "median_nonexposed_s_per_km"
            ]
        )
    ].copy()

    candidate_pairs[
        "exceedance_median_s_per_km"
    ] = (
        candidate_pairs[
            "distance_normalised_rt_s_per_km"
        ]
        - candidate_pairs[
            "median_nonexposed_s_per_km"
        ]
    )

    p75_pairs = candidate_pairs[
        candidate_pairs[
            "p75_nonexposed_s_per_km"
        ].notna()
        & (
            candidate_pairs[
                "distance_normalised_rt_s_per_km"
            ]
            > candidate_pairs[
                "p75_nonexposed_s_per_km"
            ]
        )
    ].copy()

    eligible_municipalities = set(
        municipality_stats.loc[
            municipality_stats[
                "n_exposed"
            ].fillna(0) >= 5,
            "KOM_NAMN",
        ]
    )

    min5_pairs = candidate_pairs[
        candidate_pairs[
            "KOM_NAMN"
        ].isin(
            eligible_municipalities
        )
    ].copy()

    median_counts = (
        candidate_pairs
        .groupby("KOM_NAMN")[
            "call_id"
        ]
        .nunique()
        .rename(
            "n_candidates_median"
        )
    )

    p75_counts = (
        p75_pairs
        .groupby("KOM_NAMN")[
            "call_id"
        ]
        .nunique()
        .rename(
            "n_candidates_p75"
        )
    )

    mean_exceedance = (
        candidate_pairs
        .groupby("KOM_NAMN")[
            "exceedance_median_s_per_km"
        ]
        .mean()
        .rename(
            "mean_exceedance_median_s_per_km"
        )
    )

    municipality_stats = (
        municipality_stats
        .set_index("KOM_NAMN")
        .join(median_counts)
        .join(p75_counts)
        .join(mean_exceedance)
        .reset_index()
    )

    count_columns = [
        "n_exposed",
        "n_nonexposed",
        "n_candidates_median",
        "n_candidates_p75",
    ]

    municipality_stats[
        count_columns
    ] = municipality_stats[
        count_columns
    ].fillna(0)

    length_quartiles = (
        phase2
        .groupby(
            "length_quartile",
            observed=False,
        )
        .agg(
            n_routes=("call_id", "size"),
            n_exposed=("exposed_flag", "sum"),
            median_route_length_km=(
                "route_length_km",
                "median",
            ),
        )
        .reset_index()
    )

    length_quartiles[
        "exposure_prevalence_pct"
    ] = (
        100.0
        * length_quartiles["n_exposed"]
        / length_quartiles["n_routes"]
    )

    route_exceedance = (
        candidate_pairs
        .groupby("call_id")[
            "exceedance_median_s_per_km"
        ]
        .agg(
            mean_exceedance_median_s_per_km="mean",
            max_exceedance_median_s_per_km="max",
        )
        .reset_index()
    )

    candidate_routes = phase2[
        phase2["call_id"].isin(
            candidate_pairs[
                "call_id"
            ].unique()
        )
    ].merge(
        route_exceedance,
        on="call_id",
        how="left",
    )

    phase2.drop(
        columns="geometry"
    ).to_csv(
        PHASE2_ROUTES_CSV,
        index=False,
    )

    candidate_pairs[
        [
            "call_id",
            "KOM_NAMN",
            "n_exposed",
            "n_nonexposed",
            "median_nonexposed_s_per_km",
            "p75_nonexposed_s_per_km",
            "distance_normalised_rt_s_per_km",
            "exceedance_median_s_per_km",
        ]
    ].to_csv(
        CANDIDATE_PAIRS_CSV,
        index=False,
    )

    municipality_stats.to_csv(
        MUNICIPALITY_STATS_CSV,
        index=False,
    )

    length_quartiles.to_csv(
        LENGTH_QUARTILES_CSV,
        index=False,
    )

    candidate_routes[
        [
            "call_id",
            "status_T",
            "response_time_sec",
            "route_length_m",
            "route_length_km",
            "distance_normalised_rt_s_per_km",
            "exposure_fraction",
            "mean_exceedance_median_s_per_km",
            "max_exceedance_median_s_per_km",
            "geometry",
        ]
    ].to_file(
        CANDIDATE_ROUTES_GPKG,
        driver="GPKG",
    )

    median_route_ids = candidate_pairs[
        "call_id"
    ].unique()

    p75_route_ids = p75_pairs[
        "call_id"
    ].unique()

    min5_route_ids = min5_pairs[
        "call_id"
    ].unique()

    exposed = phase2[
        phase2["exposed_flag"]
    ]

    nonexposed_phase2 = phase2[
        ~phase2["exposed_flag"]
    ]

    summary = [
        f"Routes analysed in {OPERATIONAL_YEAR}: {len(phase2)}",
        f"Exposed routes: {len(exposed)}",
        f"Exposed share (%): {100 * len(exposed) / len(phase2):.3f}",
        f"Median-screen candidate routes: {len(median_route_ids)}",
        f"Median-screen municipalities: {candidate_pairs['KOM_NAMN'].nunique()}",
        f"Mean exceedance above municipal median (s/km): {route_exceedance['mean_exceedance_median_s_per_km'].mean():.3f}",
        f"Median exceedance above municipal median (s/km): {route_exceedance['mean_exceedance_median_s_per_km'].median():.3f}",
        f"75th-percentile candidate routes: {len(p75_route_ids)}",
        f"75th-percentile municipalities: {p75_pairs['KOM_NAMN'].nunique()}",
        f"Minimum-five-exposed candidate routes: {len(min5_route_ids)}",
        f"Minimum-five-exposed municipalities: {min5_pairs['KOM_NAMN'].nunique()}",
        f"Median route length, non-exposed (km): {nonexposed_phase2['route_length_km'].median():.3f}",
        f"Median route length, exposed (km): {exposed['route_length_km'].median():.3f}",
        f"Mean exposure fraction among exposed routes: {exposed['exposure_fraction'].mean():.6f}",
    ]

    SUMMARY_TXT.write_text(
        "\n".join(summary) + "\n"
    )

    print(
        "\n".join(summary)
    )

    return (
        phase2,
        candidate_pairs,
        candidate_routes,
    )


# Run analysis

def main():
    ROUTING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calls = load_calls()

    (
        dates,
        lat_values,
        lon_values,
        dfi_values,
        date_lookup,
    ) = load_dfi()

    if RERUN_ROUTING or not ROUTES_GPKG.exists():
        graph = load_or_build_graph()
        routes = route_calls(
            calls,
            graph,
        )
    else:
        routes = gpd.read_file(
            ROUTES_GPKG
        )
        routes["status_T"] = pd.to_datetime(
            routes["status_T"],
            errors="coerce",
        )

    routes = historical_exposure(
        routes,
        dates,
        lat_values,
        lon_values,
        dfi_values,
    )

    run_phase2(
        routes,
        date_lookup,
        lat_values,
        lon_values,
        dfi_values,
    )


if __name__ == "__main__":
    main()
