from typing import Any

import torch
from class_resolver import ClassResolver
from torch import nn

activation_resolver = ClassResolver(
    [nn.PReLU, nn.Sigmoid],
    base=nn.Module,
    default=nn.PReLU,
)


class ConfigParameterError(ValueError):
    """Raised when the configuration parameters for model layers are inconsistent."""

    def __init__(self, message: str = "Please ensure the correct number of each parameter have been set") -> None:
        """
        Initialize ConfigParameterError with an optional message.

        Args:
            message (str): The error message to display.

        """
        super().__init__(message)


def weights_init(m: nn.Module) -> None:
    """
    Initialize the weights of convolutional layers using Xavier uniform initialization.

    Args:
        m (nn.Module): The module to initialize.

    """
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class EEGCNNSubject(nn.Module):
    """Model to classify to which subject the EEG belongs."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the EEG_CNN_Subject model.

        Args:
            config (dict[str, Any]): Model configuration dictionary.

        """
        super().__init__()
        self.config = config

        # Check the layer build lists are all the same size
        if not (
            self.config["layers"]["number"]
            == len(self.config["layers"]["in_channels"])
            == len(self.config["layers"]["out_channels"])
            == len(self.config["layers"]["kernel_sizes"])
            == len(self.config["layers"]["strides"])
        ):
            raise ConfigParameterError

        # Iterate over the lists and build the conv layers
        layers: list[nn.Module] = []
        for in_features, out_features, kernel_size, stride in zip(
            self.config["layers"]["in_channels"],
            self.config["layers"]["out_channels"],
            self.config["layers"]["kernel_sizes"],
            self.config["layers"]["strides"],
            strict=False,
        ):
            layers.extend(
                (
                    nn.Conv1d(
                        in_channels=in_features,
                        out_channels=out_features,
                        kernel_size=kernel_size,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(num_features=out_features),
                    nn.PReLU(),
                    nn.Dropout(self.config["dropout_level"]),
                ),
            )
        self.conv_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.config["num_class_units"], self.config["num_subjects"])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute a forward pass through the subject classification model.

        Args:
            x (torch.FloatTensor): The raw input EEG data.

        Returns:
            torch.FloatTensor: Subject class predictions for the input dataset.

        """
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)

        return self.classifier(out)


class EEGCNNGenerator(nn.Module):
    """Model to generate synthetic EEG signals."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the EEG_CNN_Generator model.

        Args:
            config (dict[str, Any]): Model configuration dictionary.

        """
        super().__init__()
        self.config = config

        # Check the layer build lists are all the same size
        if not (
            self.config["layers_gen"]["number"]
            == len(self.config["layers_gen"]["in_channels"])
            == len(self.config["layers_gen"]["out_channels"])
            == len(self.config["layers_gen"]["kernel_sizes"])
            == len(self.config["layers_gen"]["strides"])
            == len(self.config["layers_gen"]["activations"])
        ):
            raise ConfigParameterError

        self.dense = nn.Sequential(nn.Linear(103, 2816), nn.PReLU())

        # Iterate over the lists and build the conv layers
        layers: list[nn.Module] = []
        for in_features, out_features, kernel_size, stride, activation in zip(
            self.config["layers_gen"]["in_channels"],
            self.config["layers_gen"]["out_channels"],
            self.config["layers_gen"]["kernel_sizes"],
            self.config["layers_gen"]["strides"],
            self.config["layers_gen"]["activations"],
            strict=False,
        ):
            layers.extend(
                (
                    nn.ConvTranspose1d(
                        in_channels=in_features,
                        out_channels=out_features,
                        kernel_size=kernel_size,
                        stride=stride,
                        bias=False,
                    ),
                    activation_resolver.make(activation),
                ),
            )
        self.convTranspose_layers = nn.Sequential(*layers)

    def forward(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute a forward pass through the generator model.

        Args:
            z (torch.FloatTensor): The random noise to be passed through the model.

        Returns:
            torch.FloatTensor: Generated data.

        """
        out = self.dense(z)
        out = out.view(out.size(0), 16, 176)

        return self.convTranspose_layers(out)


class EEGCNNDiscriminator(nn.Module):
    """Model to classify real EEG signals from synthetic EEG signals."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the EEG_CNN_Discriminator model.

        Args:
            config (dict[str, Any]): Model configuration dictionary.

        """
        super().__init__()
        self.config = config

        # Check the layer build lists are all the same size
        if not (
            self.config["layers_dis"]["number"]
            == len(self.config["layers_dis"]["in_channels"])
            == len(self.config["layers_dis"]["out_channels"])
            == len(self.config["layers_dis"]["kernel_sizes"])
            == len(self.config["layers_dis"]["strides"])
        ):
            raise ConfigParameterError

        # Iterate over the lists and build the conv layers
        layers: list[nn.Module] = []
        for in_features, out_features, kernel_size, stride in zip(
            self.config["layers_dis"]["in_channels"],
            self.config["layers_dis"]["out_channels"],
            self.config["layers_dis"]["kernel_sizes"],
            self.config["layers_dis"]["strides"],
            strict=False,
        ):
            layers.extend(
                (
                    nn.Conv1d(
                        in_channels=in_features,
                        out_channels=out_features,
                        kernel_size=kernel_size,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(num_features=out_features),
                    nn.PReLU(),
                    nn.Dropout(self.config["dropout_level"]),
                ),
            )
        self.conv_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.config["num_class_units"], 1)
        self.aux = nn.Linear(self.config["num_class_units"], self.config["num_aux_class"])

    def forward(self, x: torch.FloatTensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Compute a forward pass through the discriminator model.

        Args:
            x (torch.FloatTensor): The input EEG data (real or synthetic).

        Returns:
            tuple[torch.FloatTensor, torch.FloatTensor]:
                - Real/fake prediction (binary classification)
                - Auxiliary class predictions

        """
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        realfake = self.classifier(out)
        classes = self.aux(out)
        return realfake, classes


class EEGCNNSSVEP(nn.Module):
    """Model to classify frequency classes from EEG signals."""

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize the EEG_CNN_SSVEP model.

        Args:
            config (dict[str, Any]): Model configuration dictionary.

        """
        super().__init__()
        self.config = config

        # Check the layer build lists are all the same size
        if not (
            self.config["layers"]["number"]
            == len(self.config["layers"]["in_channels"])
            == len(self.config["layers"]["out_channels"])
            == len(self.config["layers"]["kernel_sizes"])
            == len(self.config["layers"]["strides"])
        ):
            raise ConfigParameterError

        # Iterate over the lists and build the conv layers
        layers: list[nn.Module] = []
        for in_features, out_features, kernel_size, stride in zip(
            self.config["layers"]["in_channels"],
            self.config["layers"]["out_channels"],
            self.config["layers"]["kernel_sizes"],
            self.config["layers"]["strides"],
            strict=False,
        ):
            layers.extend(
                (
                    nn.Conv1d(
                        in_channels=in_features,
                        out_channels=out_features,
                        kernel_size=kernel_size,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm1d(num_features=out_features),
                    nn.PReLU(),
                    nn.Dropout(self.config["dropout_level"]),
                ),
            )
        self.conv_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(self.config["num_class_units"], self.config["num_class"])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute a forward pass through the SSVEP classification model.

        Args:
            x (torch.FloatTensor): The raw input EEG data.

        Returns:
            torch.FloatTensor: Frequency class predictions for the input dataset.

        """
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)

        return self.classifier(out)
