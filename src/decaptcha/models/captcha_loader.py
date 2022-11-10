"""Loading Captcha images."""

# Standard Library
from pathlib import Path
from typing import Literal

# Thirdparty Library
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms  # type: ignore
from torchvision.datasets import ImageFolder

# Package Library
from decaptcha.utils.utils import download_recaptcha_datasets, package_path


class CaptchaLoader:
    """Load ReCaptcha images."""

    def __init__(
        self,
        split_size: float = 0.8,
        batch_size: int = 10,
    ) -> None:
        """
        Initialize class instance.

        Parameters
        ----------
        split_size : float, optional
            Size of the train/test split, by default 0.8.
        batch_size : int, optional
            Number of batches to return, by default 10.
        """
        self.split_size: float = split_size
        self.batch_size: int = batch_size
        self.device: Literal["cuda", "cpu"] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def image_folder_path(self) -> str:
        """
        Path to the image datafolder.

        Returns
        -------
        str
            Path to the image datafolder.

        Notes
        -----
        The package will download the ReCaptcha dataset if it cannot be found.
        """
        recaptcha_path: Path = package_path(
            dir_="recaptcha", create_dir=False
        )

        if not recaptcha_path.exists():
            download_recaptcha_datasets()

        return str(recaptcha_path)

    def load_data(
        self,
    ) -> tuple[DataLoader[Tensor], DataLoader[Tensor]]:
        """
        Get train and test loader for the captcha images.

        Returns
        -------
        train_loader : DataLoader[Tensor]
            DataLoader for the train images.
        test_loader : DataLoader[Tensor]
            DataLoader for the train images.

        """
        assert 0 < self.split_size < 1
        assert isinstance(self.batch_size, int)

        if self.device == "cuda":
            pin_memory: bool = True
            num_workers: int = 0
        else:
            pin_memory: bool = False
            num_workers: int = 1

        image_data: ImageFolder = ImageFolder(
            root=self.image_folder_path,
            transform=transforms.Compose(  # type: ignore
                [
                    transforms.Resize(120),
                    transforms.ToTensor(),
                ]
            ),
        )

        train_size: int = int(self.split_size * len(image_data))
        test_size: int = len(image_data) - train_size

        image_dataset: list[Subset[Tensor]] = random_split(  # type: ignore
            dataset=image_data,
            lengths=[train_size, test_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_data: Subset[Tensor] = image_dataset[0]
        test_data: Subset[Tensor] = image_dataset[0]

        train_loader: DataLoader[Tensor] = DataLoader(
            train_data,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=False,
            batch_size=self.batch_size,
        )

        test_loader: DataLoader[Tensor] = DataLoader(
            test_data,
            pin_memory=pin_memory,
            num_workers=num_workers,
            shuffle=False,
            batch_size=self.batch_size,
        )

        return train_loader, test_loader
