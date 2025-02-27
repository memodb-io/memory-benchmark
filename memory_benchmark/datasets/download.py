import os
import requests
from rich.progress import track
from hashlib import sha256
from ..env import HOME_PATH, console
from .types import RemoteFile

FILES = {
    "locomo": [
        RemoteFile(
            url="https://github.com/snap-research/locomo/raw/refs/heads/main/data/locomo10.json",
            name="locomo10.json",
            hash="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
        )
    ]
}


def local_files(dataset: str):
    assert dataset in FILES, f"Dataset {dataset} not found in {list(FILES.keys())}"
    files = {}
    for df in FILES[dataset]:
        files[df.name] = os.path.join(HOME_PATH, "datasets", dataset, df.name)
    return files


def exist_or_download(dataset: str):
    if check_local_dataset_exist(dataset):
        console.log(f"Dataset {dataset} already exists")
    else:
        download_from_github(dataset)


def check_local_dataset_exist(dataset: str) -> bool:
    assert dataset in FILES, f"Dataset {dataset} not found in {list(FILES.keys())}"

    dataset_path = os.path.join(HOME_PATH, "datasets")
    if not os.path.exists(dataset_path):
        return False

    local_path = os.path.join(dataset_path, dataset)
    if not os.path.exists(local_path):
        return False

    for df in FILES[dataset]:
        local_file = os.path.join(local_path, df.name)
        if not os.path.exists(local_file):
            return False
        with open(local_file, "rb") as f:
            file_hash = sha256(f.read()).hexdigest()
            if file_hash != df.hash:
                return False
    return True


def download_from_github(dataset: str):
    assert dataset in FILES, f"Dataset {dataset} not found in {list(FILES.keys())}"

    dataset_path = os.path.join(HOME_PATH, "datasets")
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    local_path = os.path.join(dataset_path, dataset)
    if not os.path.exists(local_path):
        os.makedirs(local_path)

    # download the file
    console.log(f"Downloading {dataset} from GitHub")
    for df in track(
        FILES[dataset],
        description=f"Downloading {len(FILES[dataset])} files...",
    ):
        local_file = os.path.join(local_path, df.name)
        response = requests.get(df.url)
        response.raise_for_status()

        file_hash = sha256(response.content).hexdigest()
        assert (
            file_hash == df.hash
        ), f"Hash of {df.name}, {dataset} does not match expected hash"

        with open(local_file, "wb") as f:
            f.write(response.content)
