from pathlib import Path
import re
import urllib.parse
import logging
import os

import boto3
import botocore.client
import requests
from huggingface_hub import snapshot_download
from dotenv import dotenv_values, find_dotenv

from molmobot_pi0.utils import tqdm


CACHE_DIR = Path(os.getenv("MOLMOBOT_PI0_CACHE_DIR") or (Path.home() / ".cache" / "molmobot_pi0"))

logger = logging.getLogger(__name__)

def download_s3(
    bucket: str,
    prefix: str,
    local_path: Path,
    progbar: bool = True,
    exclude: str | None = None,
    include: str | None = None,
    **kwargs,
):
    dotenv_path = find_dotenv()
    env_values = dotenv_values(dotenv_path) if dotenv_path else {}

    # first check .env, then fall back to env variable, then fall back to anonymous
    aws_access_key_id = env_values.get("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID"))
    aws_secret_access_key = env_values.get("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY"))
    if aws_access_key_id and aws_secret_access_key:
        aws_config = None
    else:
        aws_config = botocore.client.Config(signature_version=botocore.UNSIGNED)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        config=aws_config,
    )

    # List all objects under the prefix (handles both files and directories)
    logger.info(f"Downloading {bucket}/{prefix} to {local_path}")
    paginator = s3.get_paginator("list_objects_v2")

    total_size = 0
    keys_to_download = []
    exclude_regex = re.compile(exclude) if exclude else None
    include_regex = re.compile(include) if include else None
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if exclude_regex and exclude_regex.search(key) and (not include_regex or not include_regex.search(key)):
                continue
            total_size += obj["Size"]
            keys_to_download.append(key)

    if len(keys_to_download) == 0:
        raise FileNotFoundError(f"No objects found at {bucket}/{prefix}")

    with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, disable=not progbar) as pbar:
        for key in keys_to_download:
            # handle blacklist and whitelist
            if key == prefix:
                # Single file case: download directly to local_path
                dst_path = local_path
            else:
                # Directory case: preserve relative structure under local_path
                rel_path = Path(key).relative_to(prefix)
                dst_path = local_path / rel_path

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dst_path), Callback=pbar.update)


def download_http(url: str, local_path: Path):
    logger.info(f"Downloading {url} to {local_path}")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url)
    response.raise_for_status()
    local_path.write_bytes(response.content)


def download_hf(repo_id: str, local_path: Path):
    snapshot_download(repo_id, local_dir=local_path)


def maybe_download(url: str, **kwargs) -> Path:
    parsed = urllib.parse.urlparse(url)

    if parsed.scheme == "":
        path = Path(url)
        if not path.exists():
            raise FileNotFoundError(f"File not found at {url}")
        return path

    cache_dir = CACHE_DIR.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_path = (cache_dir / parsed.netloc / parsed.path.strip("/")).resolve()
    if local_path.exists():
        return local_path

    if parsed.scheme == "s3":
        download_s3(parsed.netloc, parsed.path.lstrip("/"), local_path, **kwargs)
    elif parsed.scheme in ["http", "https"]:
        download_http(url, local_path)
    elif parsed.scheme == "hf":
        download_hf(parsed.netloc + parsed.path, local_path)

    return local_path
