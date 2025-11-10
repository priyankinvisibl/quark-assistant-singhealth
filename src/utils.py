import uuid
from datetime import UTC, datetime
from typing import Any

import boto3
from fastapi.routing import APIRoute


def get_zulu_time():
    return str(
        datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"
    )


def generate_uuid(route: APIRoute):
    return f"{route.tags[0]}_{uuid.uuid4()}"


def time_greater_than(t1, t2):
    dt1 = datetime.strptime(t1, "%Y-%m-%dT%H:%M:%SZ")
    dt2 = datetime.strptime(t2, "%Y-%m-%dT%H:%M:%SZ")
    return dt1 > dt2


def get_boto3_creds(config: dict[str, Any]) -> dict[str, str]:
    sts = {}
    cross_account_settings = config.get("values", {}).get("crossAccount")
    if cross_account_settings is not None:
        cross_account_id = cross_account_settings.get("id")
        cross_account_role = cross_account_settings.get("role")
        if cross_account_id is None or cross_account_role is None:
            raise Exception("Cross-account set in config without ID or role")

        sts_connection = boto3.client("sts")
        sts = sts_connection.assume_role(
            RoleArn=(f"arn:aws:iam::{cross_account_id}:role/{cross_account_role}"),
            RoleSessionName="quark-assistant",
        )
        return {
            "aws_access_key_id": sts["Credentials"]["AccessKeyId"],
            "aws_secret_access_key": sts["Credentials"]["SecretAccessKey"],
            "aws_session_token": sts["Credentials"]["SessionToken"],
        }
    return sts


def read_yaml(path: str):
    with open(path, "r") as f:
        import yaml

        yaml_content = yaml.safe_load(f)
    return yaml_content
