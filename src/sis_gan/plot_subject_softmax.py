import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SoftmaxSubject:
    """Class to evaluate and plot softmax probabilities for generated EEG data using a pretrained subject classifier."""

    def __init__(self) -> None:
        """Initialize SoftmaxClass by loading generated data and labels."""
        self.input_data: np.ndarray = np.load("SISGAN_unseen0.npy")
        self.input_label: np.ndarray = np.load("SISGAN_unseen0_labels.npy")
        self.mean_outputs: list[np.ndarray] = []

    def _load_pretrain_model(self) -> None:
        """Load the pretrained subject classification model."""
        self.subject_predictor: torch.nn.Module = torch.load(
            "pretrain_subject_unseen0.pt",
            map_location=device,
            weights_only=False,
        )

    def _evaluate_model(self) -> None:
        """Test the model on the test dataset and compute mean softmax probabilities."""
        self.subject_predictor.eval()
        with torch.no_grad():
            for _, data in enumerate(self.testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()

                outputs = self.subject_predictor(inputs)
                outputs = torch.softmax(outputs, dim=1)
                outputs = outputs.detach().cpu().numpy()
                self.mean_outputs.append(outputs)

        self.mean_outputs = np.concatenate(np.array(self.mean_outputs))
        self.mean_outputs = np.mean(self.mean_outputs, 0)

    def _plot_softmax(self) -> None:
        """Plot and save the mean softmax probabilities as a bar chart."""
        x = np.arange(len(self.mean_outputs))
        plt.bar(x, self.mean_outputs, color="tab:blue", width=0.4)
        plt.rc("ytick", labelsize=14)
        plt.xlabel("Subject", fontsize=15)
        plt.ylabel("Probability", fontsize=15)
        plt.savefig("softmax.pdf")

    def perform_softmax(self) -> None:
        """Calculate and plot the softmax probability on generated data using the pretrained subject weights."""
        fake_data = torch.from_numpy(self.input_data)
        fake_label = torch.from_numpy(self.input_label)
        self.testloader: DataLoader = DataLoader(
            dataset=TensorDataset(fake_data, fake_label),
            num_workers=0,
        )

        self._load_pretrain_model()
        self._evaluate_model()
        self._plot_softmax()


if __name__ == "__main__":
    trainer = SoftmaxSubject()
    trainer.perform_softmax()
