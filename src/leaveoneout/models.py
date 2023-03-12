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
    

class EEG_CNN_Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.nz = nz
        self.dense = nn.Sequential(nn.Linear(self.nz, 2816), nn.PReLU())

        self.layer1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=16, out_channels=256, kernel_size=20, stride=2, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=10, stride=2, bias=False),
            nn.PReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=5, stride=2, bias=False),
            nn.PReLU(),
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=1, bias=False),
            nn.PReLU(),
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=32, out_channels=2, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z):
        out = self.dense(z)
        out = out.view(out.size(0), 16, 176)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out


class EEG_CNN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=20, stride=4, bias=False),
            nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            nn.Dropout(dropout_level),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=2, bias=False),
            nn.BatchNorm1d(num_features=32),
            nn.PReLU(),
            nn.Dropout(dropout_level),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.PReLU(),
            nn.Dropout(dropout_level),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.PReLU(),
            nn.Dropout(dropout_level),
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=4, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(),
            nn.Dropout(dropout_level),
        )

        self.classifier = nn.Linear(2816, 1)
        self.aux = nn.Linear(2816, 3)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        realfake = self.classifier(out)
        classes = self.aux(out)

        return realfake, classes
