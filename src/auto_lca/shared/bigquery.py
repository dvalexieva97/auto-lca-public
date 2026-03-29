from google.cloud import bigquery
from datetime import datetime
from typing import Optional, List, get_type_hints, Dict, get_origin, get_args, Union
from dataclasses import fields, is_dataclass
from auto_lca.models.paper import Paper, PaperBatch
from auto_lca.models.nlp import DocumentRank
import enum

# TODO move creation elsewhere and so paper dependency

# TODO Config dataset everywhere, make class and dynamically fetch
PROJECT_ID = "ist-lca"
DATASET_NAME = "scrape"


TABLE_CREATION_MAPPING = {
    "paper": {
        "description": "Paper metadata",
        "model": Paper,
        "exclude_fields": [
            "full_text",
            "summary",
            "topics",
            "pdf_size_mb",
            "blob_url",
        ],
    },
    "paper_nlp": {
        "description": "Paper metadata",
        "model": Paper,
        "include_fields": [
            "pid",
            "inserted_at",
            "request_id",
            "blob_url",
            "error",
            "full_text",
            "summary",
            "topics",
            "pdf_size_mb",
        ],
    },
    "paper_batch_nlp": {
        "description": "NLP processing results for a batch of papers",
        "model": PaperBatch,
        "include_fields": ["inserted_at", "request_id", "topics", "ranks"],
    },
    "paper_batch_qa": {
        "description": "Question-answering results for a batch of papers",
        "model": DocumentRank,
    },
}


def get_bigquery_schema(model):
    """Dynamically generate BigQuery schema from a dataclass model, supporting ARRAYs, STRUCTs, nested dataclasses, and enums."""
    schema = []
    type_mapping = {
        str: "STRING",
        int: "INTEGER",
        float: "FLOAT64",
        bool: "BOOLEAN",
        datetime: "TIMESTAMP",
    }

    for field in fields(model):
        field_name = field.name
        resolved_type = get_type_hints(model).get(field_name, field.type)

        # --- Unwrap all Optionals and Unions with None ---
        actual_type = resolved_type
        while True:
            origin = get_origin(actual_type)
            args = get_args(actual_type)
            # Optional[T] is Union[T, NoneType]
            if origin is Union and type(None) in args:
                non_none_args = [a for a in args if a is not type(None)]
                if non_none_args:
                    actual_type = non_none_args[0]
                    continue
            break

        origin = get_origin(actual_type)
        args = get_args(actual_type)

        # Handle List[str] and List[enum] and List[dataclass]
        if origin is list and args:
            item_type = args[0]
            # Unwrap Optional in list item
            item_origin = get_origin(item_type)
            item_args = get_args(item_type)
            if item_origin is Union and type(None) in item_args:
                non_none_args = [a for a in item_args if a is not type(None)]
                if non_none_args:
                    item_type = non_none_args[0]
            # List[str]
            if item_type == str:
                schema.append(
                    bigquery.SchemaField(field_name, "STRING", mode="REPEATED")
                )
            # List[enum]
            elif isinstance(item_type, type) and issubclass(item_type, enum.Enum):
                schema.append(
                    bigquery.SchemaField(field_name, "STRING", mode="REPEATED")
                )
            # List[dataclass]
            elif is_dataclass(item_type):
                nested_fields = get_bigquery_schema(item_type)
                schema.append(
                    bigquery.SchemaField(
                        field_name, "RECORD", mode="REPEATED", fields=nested_fields
                    )
                )
            else:
                print(
                    f"⚠️ Warning: Unsupported list item type {item_type} for field '{field_name}', defaulting to STRING"
                )
                schema.append(
                    bigquery.SchemaField(field_name, "STRING", mode="REPEATED")
                )
        # Handle Dict[str, str]
        elif origin is dict and args and args[0] == str and args[1] == str:
            struct_fields = [
                # TODO Dehardcode
                bigquery.SchemaField("url", "STRING"),
                bigquery.SchemaField("status", "STRING"),
            ]
            schema.append(
                bigquery.SchemaField(
                    field_name, "RECORD", mode="NULLABLE", fields=struct_fields
                )
            )
        # Handle nested dataclasses
        elif is_dataclass(actual_type):
            nested_fields = get_bigquery_schema(actual_type)
            schema.append(
                bigquery.SchemaField(
                    field_name, "RECORD", mode="NULLABLE", fields=nested_fields
                )
            )
        # Handle enums
        elif isinstance(actual_type, type) and issubclass(actual_type, enum.Enum):
            schema.append(bigquery.SchemaField(field_name, "STRING"))
        # Handle standard types
        elif actual_type in type_mapping:
            schema.append(bigquery.SchemaField(field_name, type_mapping[actual_type]))
        else:
            print(
                f"⚠️ Warning: Unsupported field type {resolved_type} for field '{field_name}', defaulting to STRING"
            )
            schema.append(
                bigquery.SchemaField(field_name, "STRING")
            )  # Default to STRING

    return schema


def filter_columns(
    row, columns: List[str], inserted_at
) -> Dict[str, Optional[Union[str, int, float]]]:
    """Filters the row to only include specified columns."""
    row["inserted_at"] = inserted_at
    res = {k: row.get(k) for k in columns if k in row}
    print("Row for insert:", res)
    return res


def query_table(query_template: str, params: Dict[str, Union[str, int, float]]):
    """
    Executes a parameterized query on the BigQuery dataset and returns the results.

    Args:
        query_template: SQL query string with named parameters (e.g., @param).
        params: Dictionary of parameter names to values.

    Returns:
        List of rows from the query result.
    """
    bq_client = bigquery.Client()
    query_parameters = []
    for name, value in params.items():
        # Infer type for BigQuery parameter
        if isinstance(value, str):
            param_type = "STRING"
        elif isinstance(value, int):
            param_type = "INT64"
        elif isinstance(value, float):
            param_type = "FLOAT64"
        else:
            param_type = "STRING"
        query_parameters.append(bigquery.ScalarQueryParameter(name, param_type, value))

    job_config = bigquery.QueryJobConfig(query_parameters=query_parameters)
    query_job = bq_client.query(query_template, job_config=job_config)
    rows = list(query_job)
    return rows


def format_query(query: str, params) -> str:
    """
    Formats the query string with parameters for better readability.
    """
    subquery_clause = params.pop("__SUBQUERY__", None)
    for key, value in params.items():
        if isinstance(value, str):
            value = f"'{value}'"
        # TODO improve logic
        query = query.replace(f"@{key}", str(value))
    if subquery_clause:
        subquery_str = " AND ".join(
            # TODO handle types
            f"{k} = '{v}'"
            for k, v in subquery_clause.items()
            if v is not None
        )
    else:
        subquery_str = "1=1"

    query = query.replace(f"@__SUBQUERY__", subquery_str)

    return query


def save_results_to_bigquery(data, table_name: str):
    """Saves all paper metadata (including errors) to BigQuery."""
    try:
        if not data:
            print("❌ No data to insert.")
            return None

        client = bigquery.Client()
        table_id = f"{PROJECT_ID}.{DATASET_NAME}.{table_name}"
        try:
            schema = client.get_table(table_id).schema  # TODO check if slow
            columns = [field.name for field in schema]
        except:
            columns = list(data[0].keys())

        print("COLS:", columns)
        inserted_at = str(datetime.now())
        data = [filter_columns(row, columns, inserted_at) for row in data]
        errors = client.insert_rows_json(table_id, data)

        if errors:
            unique_errors = {error["errors"][0]["message"] for error in errors}
            print(f"❌ BigQuery Insert Errors: {unique_errors}")  # TODO Log
        else:
            print(f"✅ Successfully inserted {len(data)} records into BigQuery.")
        return errors
    except Exception as e:
        print(f"❌ Error inserting into BigQuery: {e}")


def create_bigquery_dataset():
    """Creates the BigQuery dataset if it does not exist."""
    client = bigquery.Client()
    dataset_id = f"{PROJECT_ID}.{DATASET_NAME}"

    dataset = bigquery.Dataset(dataset_id)
    dataset.location = "US"  # Set dataset location

    try:
        client.create_dataset(dataset, exists_ok=True)
        print(f"✅ Dataset created or already exists: {dataset_id}")
    except Exception as e:
        print(f"❌ Error creating dataset: {e}")


def create_bigquery_tables():
    """Creates the BigQuery table dynamically based on the Paper model."""
    client = bigquery.Client()
    dataset_id = f"{PROJECT_ID}.{DATASET_NAME}"
    create_bigquery_dataset()

    for table_name, dict in TABLE_CREATION_MAPPING.items():
        table_id = f"{dataset_id}.{table_name}"
        schema_obj = dict["model"]
        exclude_fields = dict.get("exclude_fields")
        include_fields = dict.get("include_fields")
        schema = get_bigquery_schema(schema_obj)
        if include_fields:
            schema = [field for field in schema if field.name in include_fields]
        if exclude_fields:
            schema = [field for field in schema if field.name not in exclude_fields]

        try:
            table = bigquery.Table(table_id, schema=schema)
            table = client.create_table(table, exists_ok=True)
            print(f"✅ BigQuery table exists or created: {table_id}")
        except Exception as e:
            print(f"❌ Error creating BigQuery table: {e}")


# create_bigquery_tables()  # TODO Add resource creation to deployment
