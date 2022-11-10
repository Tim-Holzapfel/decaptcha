"""Main package model."""

# Standard Library
from functools import partial
from typing import Any, Literal

# Thirdparty Library
import torch
from torch import nn


class Conv2dAuto(nn.Conv2d):
    """Convolution layer."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)


class VGGNet(nn.Module):
    """
    VGG Network.

    A Visual Geometry Group (VGG) Convolutional Neural Network (CNN) for deep
    learning.
    """

    def __init__(
        self,
        spec: Literal[12, 13, 16],
        classes: int,
        kernel_size: int = 3,
        in_channels: int = 3,
    ) -> None:
        """
        Initialize class instance.

        Parameters
        ----------
        spec : Literal[12, 13, 16]
            Model specification.
        classes : int
            Number of classes to use for the model.
        kernel_size : int, optional
            Size of the kernel, by default 3.
        in_channels : int, optional
            Number of input channels of the image, by default 3.
        """
        super().__init__()
        self.spec: Literal[12, 13, 16] = spec
        self.classes: int = classes
        self.kernel_size: int = kernel_size
        self.in_channels: int = in_channels
        self.conv3x3: partial[Conv2dAuto] = partial(
            Conv2dAuto, kernel_size=3, bias=True
        )
        self.network_layers: nn.Sequential = self.vgg_norm()

    def vgg_norm(self) -> nn.Sequential:
        """
        Network layer to normalize.

        Returns
        -------
        nn.Sequential
            Sequential container.

        Raises
        ------
        TypeError
            If the input to the convolution was not either int or str.
        """
        layers: list[nn.Module] = []
        in_channels = 3
        conv_arch: dict[int, list[int | str]] = dict(
            {
                11: [
                    64,
                    "M",
                    128,
                    "M",
                    256,
                    256,
                    "M",
                    512,
                    512,
                    "M",
                    512,
                    512,
                    "M",
                ],
                13: [
                    64,
                    64,
                    "M",
                    128,
                    128,
                    "M",
                    256,
                    256,
                    "M",
                    512,
                    512,
                    "M",
                    512,
                    512,
                    "M",
                ],
                16: [
                    64,
                    64,
                    "M",
                    128,
                    128,
                    "M",
                    256,
                    256,
                    256,
                    "M",
                    512,
                    512,
                    512,
                    "M",
                    512,
                    512,
                    512,
                    "M",
                ],
            }
        )
        for v in conv_arch[self.spec]:
            if isinstance(v, str):
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(v, int):
                layers += [
                    self.conv3x3(in_channels, v, device="cuda"),
                    nn.BatchNorm2d(v, device="cuda"),
                    nn.ReLU(),
                ]
                in_channels: int = v
            else:
                raise TypeError(
                    """The input of the convolution layer must be ither an
                    integer or a string!"""
                )
        return nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Linear(4608, 4096, device="cuda"),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, device="cuda"),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, self.classes, device="cuda"),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> Any:  # type: ignore
        """
        Forward part of the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Tensor to which the neural network will be applied.

        Returns
        -------
        output_fin : Any
            Neural network tensor.
        """
        output_fin = self.network_layers(x)
        return output_fin
