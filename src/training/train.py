import torch
from torch.utils.data import DataLoader

from src.consts import Consts
from src.utils.utils import Utils

from src.dataset.preprocessed_dataset import PreprocessedDataset
from src.modules.generator import Generator
from src.modules.discriminator import Discriminator


class CycleGanTraining:
    def __init__(self,  # only dataset parameters are given explicitly in constructor
                 A_data_file,
                 B_data_file,
                 number_of_frames,
                 batch_size,
                 A_validation_dir,
                 B_validation_dir,
                 A_output_dir,
                 B_output_dir):

        # ------------------------------ #
        #  hyper parameters              #
        # ------------------------------ #
        self.number_of_epochs = Consts.number_of_epochs
        self.batch_size = batch_size
        self.cycle_loss_lambda = Consts.cycle_loss_lambda
        self.identity_loss_lambda = Consts.identity_loss_lambda
        self.zero_identity_lambda_loss_after = Consts.zero_identity_loss_lambda_after
        self.start_decay_after = Consts.start_decay_after
        self.device = Utils.get_device()

        # ------------------------------ #
        #  dataloader                    #
        # ------------------------------ #
        dataset = CycleGanTraining._prepare_dataset(A_data_file, B_data_file, number_of_frames)
        self.dataloader = CycleGanTraining._prepare_dataloader(dataset, batch_size)
        self.number_of_frames = number_of_frames
        self.number_of_samples_in_dataset = len(dataset)

        # ------------------------------ #
        #  generators and discriminators #
        # ------------------------------ #
        self.A2B_gen = Generator().to(self.device)
        self.B2A_gen = Generator().to(self.device)
        self.A_disc = Discriminator().to(self.device)
        self.B_disc = Discriminator().to(self.device)

        # ------------------------------ #
        #  optimizers                    #
        # ------------------------------ #
        gen_params = list(self.A2B_gen.parameters()) + list(self.B2A_gen.parameters())
        disc_params = list(self.A_disc.parameters()) + list(self.B_disc.parameters())

        gen_lr = Consts.generator_lr
        disc_lr = Consts.discriminator_lr

        self.gen_optimizer = \
            torch.optim.Adam(gen_params, lr=gen_lr, betas=Consts.adam_optimizer_betas)
        self.disc_optimizer = \
            torch.optim.Adam(disc_params, lr=disc_lr, betas=Consts.adam_optimizer_betas)

        self.gen_loss_store = []
        self.disc_loss_store = []

        # ------------------------------ #
        #  validations dirs              #
        # ------------------------------ #
        self.A_validation_dir = A_validation_dir
        self.B_validation_dir = B_validation_dir
        self.A_output_dir = A_output_dir
        self.B_output_dir = B_output_dir

    def train(self):
        for epoch_num in range(self.number_of_epochs):
            print(f"Epoch {epoch_num + 1}")
            self._train_single_epoch(epoch_num)
        print("Finished training")

    def _train_single_epoch(self, epoch_num):
        for i, (real_A, real_B) in enumerate(self.dataloader):
            iteration = (self.number_of_samples_in_dataset // self.batch_size) * epoch_num + i

            if iteration > self.zero_identity_lambda_loss_after:
                self.identity_loss_lambda = 0
            if iteration > self.start_decay_after:
                pass  # TODO

            # TODO

    @staticmethod
    def _prepare_dataset(A_data_file, B_data_file, number_of_frames):
        dataset = PreprocessedDataset(
            A_dataset_file=A_data_file,
            B_dataset_file=B_data_file,
            number_of_frames=number_of_frames
        )
        return dataset

    @staticmethod
    def _prepare_dataloader(dataset, batch_size):
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        return dataloader


if __name__ == '__main__':

    cycleGanTraining = CycleGanTraining(
        A_data_file=Consts.A_preprocessed_dataset_file,
        B_data_file=Consts.B_preprocessed_dataset_file,
        number_of_frames=Consts.number_of_frames,
        batch_size=Consts.mini_batch_size,
        A_validation_dir='dupa1',
        B_validation_dir='dupa2',
        A_output_dir=Consts.A_output_dir,
        B_output_dir=Consts.B_output_dir
    )
