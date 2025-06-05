import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Softmax_Class:
    def __init__(self) -> None:
        self.input_data = np.load("SISGAN_unseen0.npy")
        self.input_label = np.load("SISGAN_unseen0_labels.npy")

    def _load_pretrain_model(self) -> None:
        """Load the pretrain subject classification model"""
        # Load the pretrain subject predictor
        self.subject_predictor = torch.load("pretrain_subject_unseen0.pt", map_location=torch.device("cuda:0"))

    def _test_model(self) -> None:
        """Test the model on test dataset"""
        self.mean_outputs = []
        self.subject_predictor.eval()
        with torch.no_grad():
            for i, data in enumerate(self.testloader, 0):
                # format the data from the dataloader
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.float()

                outputs = self.subject_predictor(inputs)
                outputs = torch.softmax(outputs, dim=1)
                outputs = outputs.detach().cpu().numpy()
                self.mean_outputs.append(outputs)

            self.mean_outputs = np.concatenate(np.array(self.mean_outputs))
            self.mean_outputs = np.mean(self.mean_outputs, 0)
            print(self.mean_outputs)

    def _plot_softmax(self) -> None:
        X = np.arange(2)

        plt.bar(X, self.mean_outputs, color="tab:blue", width=0.4)

        plt.rc("ytick", labelsize=14)

        plt.xlabel("Subject", fontsize=15)
        plt.ylabel("Probability", fontsize=15)

        plt.savefig("softmax.pdf")

    def perform_softmax(self) -> None:
        """Calculate the softmax probability on generated data based on the pretrained subject weight"""
        fake_data = torch.from_numpy(self.input_data)
        fake_label = torch.from_numpy(self.input_label)
        self.testloader = DataLoader(
            dataset=TensorDataset(fake_data, fake_label),
            num_workers=0,
        )

        self._load_pretrain_model()
        self._test_model()
        self._plot_softmax()


if __name__ == "__main__":
    trainer = Softmax_Class()
    trainer.perform_softmax()
