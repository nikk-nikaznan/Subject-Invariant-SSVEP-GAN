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
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=20, stride=4, bias=False),
            nn.BatchNorm1d(num_features=16),
            nn.PReLU(),
            nn.Dropout(self.config["dropout_level"]),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=2, bias=False),
            nn.BatchNorm1d(num_features=32),
            nn.PReLU(),
            nn.Dropout(self.config["dropout_level"]),
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(num_features=64),
            nn.PReLU(),
            nn.Dropout(self.config["dropout_level"]),
        )

        self.layer4 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm1d(num_features=128),
            nn.PReLU(),
            nn.Dropout(self.config["dropout_level"]),
        )

        self.layer5 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=2, stride=4, bias=False),
            nn.BatchNorm1d(num_features=256),
            nn.PReLU(),
            nn.Dropout(self.config["dropout_level"]),
        )

        self.classifier = nn.Linear(2816, self.config["num_subjects"])

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """The forward function to compute a pass through the subject classification model.

        Args:
            x (torch.FloatTensor): The raw input EEG data to be passed through the model.
        Returns:
            torch.FloatTensor: Class predictions for the input dataset.
        """

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out
