import torch
import torch.nn as nn
from class_resolver import ClassResolver
from itertools import chain

activation_resolver = ClassResolver(
    [nn.PReLU, nn.Sigmoid],
    base=nn.Module,
    default=nn.PReLU,
)


def weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class EEG_CNN_Subject(nn.Module):
    """Model to classify to which subject does the EEG belong."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        # Check the layer build lists are all the same size
        assert (
            self.config["layers"]["number"]
            == len(self.config["layers"]["in_channels"])
            == len(self.config["layers"]["out_channels"])
            == len(self.config["layers"]["kernel_sizes"])
            == len(self.config["layers"]["strides"])
        ), "Please ensure the correct number of each parameter have been set"

        # Iterate over the lists and build the conv layers
        layers: list = []
        for in_features, out_features, kernel_size, stride in zip(
            self.config["layers"]["in_channels"],
            self.config["layers"]["out_channels"],
            self.config["layers"]["kernel_sizes"],
            self.config["layers"]["strides"],
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
                )
            )
        self.conv_layers = nn.Sequential(*layers)

        self.classifier = nn.Linear(self.config["num_class_units"], self.config["num_subjects"])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """The forward function to compute a pass through the subject classification model.

        Args:
            x (torch.FloatTensor): The raw input EEG data to be passed through the model.
        Returns:
            torch.FloatTensor: Class predictions for the input dataset.
        """

        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


class EEG_CNN_Generator(nn.Module):
    """Model to generate synthetic EEG signals."""

    def __init__(
        self,
        config: dict,
        activation: None | str | nn.Module | type[nn.Module] = None,
        activation_kwargs: None | dict[str, any] = None,
    ) -> None:
        super().__init__()

        self.config = config
        activation_resolver.make(activation, activation_kwargs)

        # Check the layer build lists are all the same size
        assert (
            self.config["layers_gen"]["number"]
            == len(self.config["layers_gen"]["in_channels"])
            == len(self.config["layers_gen"]["out_channels"])
            == len(self.config["layers_gen"]["kernel_sizes"])
            == len(self.config["layers_gen"]["strides"])
        ), "Please ensure the correct number of each parameter have been set"

        self.dense = nn.Sequential(nn.Linear(103, 2816), nn.PReLU())

        # Iterate over the lists and build the conv layers
        layers: list = []
        for in_features, out_features, kernel_size, stride in zip(
            self.config["layers_gen"]["in_channels"],
            self.config["layers_gen"]["out_channels"],
            self.config["layers_gen"]["kernel_sizes"],
            self.config["layers_gen"]["strides"],
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
                    # nn.PReLU(),
                    activation_resolver.make(activation, activation_kwargs),
                )
            )
        self.convTranspose_layers = nn.Sequential(*layers)

        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=2, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.dense(z)
        out = out.view(out.size(0), 16, 176)
        out = self.convTranspose_layers(out)
        out = self.layer5(out)
        return out


class EEG_CNN_Discriminator(nn.Module):
    """Model to classify real EEG signals from synthetic EEG signals."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        # Check the layer build lists are all the same size
        assert (
            self.config["layers_dis"]["number"]
            == len(self.config["layers_dis"]["in_channels"])
            == len(self.config["layers_dis"]["out_channels"])
            == len(self.config["layers_dis"]["kernel_sizes"])
            == len(self.config["layers_dis"]["strides"])
        ), "Please ensure the correct number of each parameter have been set"

        # Iterate over the lists and build the conv layers
        layers: list = []
        for in_features, out_features, kernel_size, stride in zip(
            self.config["layers_dis"]["in_channels"],
            self.config["layers_dis"]["out_channels"],
            self.config["layers_dis"]["kernel_sizes"],
            self.config["layers_dis"]["strides"],
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
                )
            )
        self.conv_layers = nn.Sequential(*layers)

        self.classifier = nn.Linear(self.config["num_class_units"], 1)
        self.aux = nn.Linear(self.config["num_class_units"], self.config["num_aux_class"])

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        realfake = self.classifier(out)
        classes = self.aux(out)

        return realfake, classes


class EEG_CNN_SSVEP(nn.Module):
    """Model to classify real EEG signals from synthetic EEG signals."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config

        # Check the layer build lists are all the same size
        assert (
            self.config["layers"]["number"]
            == len(self.config["layers"]["in_channels"])
            == len(self.config["layers"]["out_channels"])
            == len(self.config["layers"]["kernel_sizes"])
            == len(self.config["layers"]["strides"])
        ), "Please ensure the correct number of each parameter have been set"

        # Iterate over the lists and build the conv layers
        layers: list = []
        for in_features, out_features, kernel_size, stride in zip(
            self.config["layers"]["in_channels"],
            self.config["layers"]["out_channels"],
            self.config["layers"]["kernel_sizes"],
            self.config["layers"]["strides"],
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
                )
            )
        self.conv_layers = nn.Sequential(*layers)

        self.classifier = nn.Linear(self.config["num_class_units"], self.config["num_class"])

    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)
        classes = self.classifier(out)

        return classes
