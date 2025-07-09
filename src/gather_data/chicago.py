import os
import time
import pandas as pd
from sodapy import Socrata
import geopandas as gpd
import requests

# Dataset and API configuration
DATASET_ID = "ijzp-q8t2"
DOMAIN = "data.cityofchicago.org"
DEFAULT_LIMIT = 100_000

GEOJSON_ENDPOINTS = {
    "districts": {
        "url": "https://data.cityofchicago.org/resource/24zt-jpfn.geojson",
        "default_filename": "police_districts.geojson"
    },
    "beats": {
        "url": "https://data.cityofchicago.org/resource/n9it-hstw.geojson",
        "default_filename": "police_beats.geojson"
    }
}


def count_records(client: Socrata, year: int | None = None) -> int:
    """
    Count the number of records for a given year.

    Args:
        client (Socrata): Socrata client instance.
        year (int, optional): Year to filter the data. Defaults to None.

    Returns:
        int: Number of records matching the filter.
    """
    where_clause = f"year = {year}" if year else None
    result = client.get(DATASET_ID, select="count(*)", where=where_clause)
    return int(result[0]['count'])


def download_chicago_crime_data(
    token_app: str | None = None,
    year: int | None = None,
    save_to_csv: bool = True,
    max_rows: int | None = None,
    output_dir: str = "",
    filename: str | None = None
) -> pd.DataFrame:
    """
    Downloads crime data from the Chicago open data portal. Automatically handles
    pagination if the number of rows exceeds the API limit.

    Args:
        token_app (str, optional): Socrata app token for authenticated requests. Defaults to None.
        year (int, optional): Year to filter the records. If None, retrieves all years. Defaults to None.
        save_to_csv (bool, optional): Whether to save the results as a CSV file. Defaults to True.
        max_rows (int, optional): If provided, limits the number of rows downloaded. Useful for testing. Defaults to None.
        output_dir (str, optional): Directory to save the CSV file. Defaults to "src/data/chicago".
        filename (str, optional): Custom filename for the CSV (without path). If None, a default name is generated.

    Returns:
        pd.DataFrame: DataFrame with the downloaded crime records.
    """
    os.makedirs(output_dir, exist_ok=True)
    client = Socrata(DOMAIN, token_app, timeout=60)
    where_clause = f"year = {year}" if year else None

    total = count_records(client, year)
    print(f"🔎 Found {total} records for year {year}.")

    # Apply max_rows limit if provided
    if max_rows is not None:
        total = min(total, max_rows)
        print(f"⚠️ Download limited to {total} rows.")

    # Download in a single request if under limit
    if total <= DEFAULT_LIMIT:
        print(f"⬇️ Downloading all in a single request ({total} rows)...")
        results = client.get(DATASET_ID, where=where_clause, limit=total)
        df = pd.DataFrame.from_records(results)
    else:
        print(f"📦 Downloading in chunks of {DEFAULT_LIMIT} rows...")
        dfs = []
        for offset in range(0, total, DEFAULT_LIMIT):
            print(f"⬇️ Downloading from offset {offset}...")
            try:
                limit = min(DEFAULT_LIMIT, total - offset)
                results = client.get(
                    DATASET_ID,
                    where=where_clause,
                    limit=limit,
                    offset=offset
                )
                df_chunk = pd.DataFrame.from_records(results)
                dfs.append(df_chunk)
            except Exception as e:
                print(f"❌ Error while downloading offset {offset}: {e}")
                break
            time.sleep(1)  # Avoid overwhelming the server

        df = pd.concat(dfs, ignore_index=True)

    # Save if requested
    if save_to_csv:
        if not filename:
            filename = f"crimes_{year}_{total}.csv" if year else f"crimes_{total}.csv"
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"✅ Data saved to: {output_path} ({len(df)} rows)")

    return df


def download_police_boundaries(which="both", output_dir="", filenames=None):
    """
    Downloads police boundaries from the City of Chicago data portal in GeoJSON format.

    Args:
        which (str): One of "districts", "beats", or "both". Determines what to download.
        output_dir (str): Path to save the files.
        filenames (dict, optional): Dictionary to override default filenames, e.g., {'beats': 'beats.geojson'}

    Returns:
        dict[str, GeoDataFrame]: Dictionary with keys "districts" and/or "beats".
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    to_download = ["districts", "beats"] if which == "both" else [which]

    for key in to_download:
        url = GEOJSON_ENDPOINTS[key]["url"]
        fname = filenames.get(key) if filenames and key in filenames else GEOJSON_ENDPOINTS[key]["default_filename"]
        output_path = os.path.join(output_dir, fname)

        print(f"⬇️ Downloading police {key} GeoJSON...")
        response = requests.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"❌ Failed to download {key}. Status: {response.status_code}")

        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"✅ {key.capitalize()} saved to: {output_path}")

        gdf = gpd.read_file(output_path)
        print(f"📍 {key.capitalize()} loaded: {len(gdf)} records")
        results[key] = gdf

    return results
