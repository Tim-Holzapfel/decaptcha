"""Main trainings loop."""

# Standard Library
import io
from pathlib import Path
from typing import Literal

# Thirdparty Library
import torch
from progressbar import progressbar
from torch import Tensor, optim
from torch.amp.autocast_mode import autocast
from torch.autograd.grad_mode import no_grad
from torch.nn.modules import CrossEntropyLoss, loss
from torch.utils.data import DataLoader

# Package Library
from decaptcha.models.captcha_loader import CaptchaLoader
from decaptcha.models.vgg_net import VGGNet as net
from decaptcha.utils.typings import Checkpoint
from decaptcha.utils.utils import package_path


class TrainModel(CaptchaLoader):
    """Train decaptcha model."""

    def __init__(
        self,
        model_name: str = "model_state",
        epochs: int = 10,
        continue_loop: bool = True,
    ) -> None:
        """
        Initialize class instance.

        Parameters
        ----------
        model_name : str, optional
            Name to save the model under, by default "model_state".
        epochs : int, optional
            Number of training epochs, by default 10.
        continue_loop : bool, optional
            Continue training of last session, by default True.
        """
        self.model_name: str = model_name
        self.epochs: int = epochs
        self.continue_loop: bool = continue_loop
        self.epoch: int = 1
        self.running_loss: float = 0
        self.correct: int = 0
        self.total: int = 0
        self.model: net = net(spec=16, classes=12)
        # Initialize the optimizer function.
        self.optimizer: optim.SGD = optim.SGD(
            self.model.parameters(), lr=0.001, momentum=0.9
        )
        # Initialize the loss function.
        self.loss_fn: loss.CrossEntropyLoss = loss.CrossEntropyLoss()
        data_loader: tuple[
            DataLoader[Tensor], DataLoader[Tensor]
        ] = self.load_data()
        self.train_loader: DataLoader[Tensor] = data_loader[0]
        self.test_loader: DataLoader[Tensor] = data_loader[1]
        self.device: Literal["cuda", "cpu"] = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def train_(
        self,
    ) -> tuple[float, float]:
        """
        Model train loop.

        Returns
        -------
        tuple[float, float]
            Train loss and train accuracy.
        """
        running_loss: float = 0
        correct: int = 0
        total: int = 0

        i: int = 0

        for imgs, labels in progressbar(
            self.train_loader, redirect_stdout=True
        ):

            i += 1

            self.model.train()

            if self.device == "cuda":
                imgs, labels = imgs.cuda(), labels.cuda()

            for param in self.model.parameters():
                param.grad = None

            self.optimizer.zero_grad()

            with autocast(device_type=self.device):
                outputs = self.model(imgs)
                loss = self.loss_fn(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if i % 100 == 0:
                train_loss: float = running_loss / len(self.train_loader)
                train_accuracy: float = 100.0 * correct / total
                print(f"Train Accuracy: {train_accuracy:.13f}")
                print(f"Train Loss: {train_loss:.13f}")

        train_loss: float = running_loss / len(self.train_loader)
        train_accuracy: float = 100.0 * correct / total

        return (train_loss, train_accuracy)

    def test_(
        self,
    ) -> tuple[float, float]:
        """
        Test Loop.

        Returns
        -------
        tuple[float, float]
            Test loss and test accuracy.
        """
        running_loss: int = 0
        correct: int = 0
        total: int = 0

        with no_grad():
            for imgs, labels in self.test_loader:

                self.model.eval()

                if self.device == "cuda":
                    imgs: torch.Tensor = imgs.cuda()
                    labels: torch.Tensor = labels.cuda()
                outputs = self.model(imgs)

                loss = self.loss_fn(outputs, labels)
                running_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss: float = running_loss / len(self.test_loader)
        val_accuracy: float = 100.0 * correct / total

        print("Test Loss: %.3f | Accuracy: %.3f" % (val_loss, val_accuracy))

        return (val_loss, val_accuracy)

    def __call__(self) -> None:
        """
        Start training and test loop.

        Returns
        -------
        None.
        """
        checkpoint_path: str = str(
            package_path(
                dir_="checkpoints", file_=self.model_name, ending_="pth"
            )
        )
        torch_buffer: io.BytesIO = io.BytesIO()

        # Load model checkpoint if the file exists
        if self.continue_loop and Path(checkpoint_path).exists():
            checkpoint: Checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch: int = checkpoint["epoch"]
            self.loss_fn: CrossEntropyLoss = checkpoint["loss"]
        else:
            # Initialize the loss function
            self.loss_fn: loss.CrossEntropyLoss = loss.CrossEntropyLoss()
            self.epoch: int = 1
        try:
            for epoch in range(1, self.epochs + 1):
                self.train_()
                self.test_()

                torch.save(self.model, torch_buffer)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": self.loss_fn,
                    },  # type: ignore
                    checkpoint_path,
                )
        except KeyboardInterrupt:

            torch.save(self.model, torch_buffer)
            torch.save(
                {
                    "epoch": self.epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": self.loss_fn,
                },  # type: ignore
                checkpoint_path,
            )
        return None
