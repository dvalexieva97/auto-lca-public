import json
from datetime import datetime
from hashlib import sha256

import pandas as pd
from google.cloud import storage


def hash_sha256(data: str) -> str:
    return sha256(data.encode()).hexdigest()


def upload_blob(bucket_name, destination_blob_name, temp_file):
    """Uploads a file to GCP"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(temp_file)
    return blob.public_url


def get_log_file_path():
    folder = "src/auto_lca/output/logs"
    time = str(datetime.now()).replace(":", "_")[:10]
    path = folder + f"{time}_log_time.csv"
    return path


def save_list_to_jsonl(data_list, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for obj in data_list:  # TODO Parallelize
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def save_list_to_json(data_list: list, save_path: str):
    with open(
        save_path,
        "w",
        encoding="utf-8",
    ) as jf:
        json.dump(data_list, jf, indent=2, ensure_ascii=False, default=str)
    return None


def flatten_extracted(extracted, keep_last_key_only=True):
    """
    Flatten list of dicts into a dataframe.
    keep_last_key_only: flag whether column name should be composed only
    of final subkey
    """
    flat_entry = {}
    for entry in extracted:
        for k, subdict in entry.items():
            for sub_key, value in subdict.items():
                if keep_last_key_only:
                    key_name = f"{sub_key}"
                else:
                    key_name = f"{k}_{sub_key}"
                flat_entry[key_name] = value
    return pd.DataFrame([flat_entry])
