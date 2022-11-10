"""Utility functions."""

# Standard Library
from pathlib import Path
from typing import Optional, cast
from zipfile import ZipFile

# Thirdparty Library
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from importlib_resources import files
from importlib_resources.abc import Traversable
from requests import Response
from tqdm import tqdm


def imshow(img: torch.Tensor) -> None:
    """
    Show image based on tensor.

    Parameters
    ----------
    img : torch.Tensor
        Tensor to show as an image.
    """
    img_mod: torch.Tensor = img / 2 + 0.5  # unnormalize
    npimg = img_mod.numpy()  # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def package_path(
    dir_: str,
    file_: Optional[str] = None,
    ending_: Optional[str] = None,
    create_dir: bool = True,
) -> Path:
    """
    Get path to package directory.

    Parameters
    ----------
    dir_ : str
        _description_
    file_ : Optional[str], optional
        Name of directory, by default None.
    ending_ : Optional[str], optional
        Ending of the file, by default None.
    create_dir : bool, optional
        Create directory if it does not exist, by default True.

    Returns
    -------
    Path
        Path to the desired files in the specified directory.
    """
    if file_ is not None and ending_ is not None:
        file_ = file_ + "." + ending_
    data_trav: Traversable = files("DeCaptcha").joinpath("data", dir_)
    data_path: Path = Path(str(data_trav))

    if not data_path.exists() and create_dir:
        data_path.mkdir(parents=True)

    if file_ is not None:
        dir_path: Path = data_path.joinpath(file_)
    else:
        dir_path = data_path

    return dir_path


def download_recaptcha_datasets() -> None:
    """
    Download and unzip the main recaptcha dataset.

    Returns
    -------
    None.
    """
    recaptcha_path = files("DeCaptcha").joinpath("data").joinpath("recaptcha")
    recaptcha_path = cast(Path, recaptcha_path)
    recaptcha_path.mkdir(parents=True, exist_ok=True)

    url: str = "https://github.com/brian-the-dev/recaptcha-dataset/archive/refs/heads/main.zip"  # type: ignore
    file_path: str = "D:/Github/DeCaptcha/src/DeCaptcha/data/main.zip"

    response: Response = requests.get(url, stream=True, verify=False)

    with open(file_path, "wb") as handle:
        for data in tqdm(response.iter_content()):  # type: ignore
            handle.write(data)  # type: ignore

    with ZipFile(file_path, mode="r") as z_file:
        for zip_name in z_file.namelist():
            zip_path: Path = Path(zip_name)
            if zip_path.suffix == ".png":
                # Folder to which the image will be extracted
                png_folder: Path = recaptcha_path.joinpath(
                    zip_path.parent.name
                )
                # Create folder of it does not already exist
                png_folder.mkdir(parents=True, exist_ok=True)

                # Path to which the image will be written
                png_path: Path = png_folder.joinpath(zip_path.name)

                with z_file.open(zip_name, mode="r") as f:
                    png_bin: bytes = f.read()

                with open(png_path, mode="wb") as fb:
                    fb.write(png_bin)

    return None
