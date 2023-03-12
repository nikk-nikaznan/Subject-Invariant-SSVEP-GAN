import torch
import torch.nn as nn


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
    