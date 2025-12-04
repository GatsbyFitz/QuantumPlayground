import pandas as pd

def load_s3_csv(s3_uri: str, storage_options: dict | None = None) -> pd.DataFrame:
    """
    Load a CSV from S3 into a pandas DataFrame.

    Example:
        df = load_s3_csv(
            "s3://nem-web-data/NEMWebScrape/Current_Operational_Demand_Actual_HH/NEMWEB_Current_Operational_Demand_Actual_HH_48_20251204154457.csv",
            storage_options={"profile": "default"}  # or {"key": "...", "secret": "..."}
        )
    """
    # storage_options can include AWS auth like:
    # {"profile": "your-aws-profile"} or {"key": "...", "secret": "...", "token": "..."}
    return pd.read_csv(s3_uri, storage_options=storage_options)



___all___ = ["load_s3_csv"]

