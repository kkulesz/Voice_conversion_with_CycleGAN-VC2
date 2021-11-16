import os
import numpy as np
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from consts import Consts
from src.data_processing.dataset import PreprocessedDataset, LightningValidationDataset
from src.data_processing.validator import Validator
from src.model.cycle_gan_vc2.generator import GeneratorCycleGan2
from src.model.cycle_gan_vc2.discriminator import DiscriminatorCycleGan2
from src.utils.files_operator import FilesOperator

"""
TODO:
    1. loading/saving model
    2. validation
        - dataset
        - pl.module
    3. proper logging
"""
class LightningCycleGan(pl.LightningModule):
    def __init__(self,
                 A_dataset,
                 B_dataset,
                 A_validation_source_dir,
                 B_validation_source_dir,
                 A2B_validation_output_dir,
                 B2A_validation_output_dir,
                 A_cache_dir,
                 B_cache_dir
                 # TODO load and save model
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.disc_lr = Consts.discriminator_lr
        self.gen_lr = Consts.generator_lr
        self.cycle_loss_lambda = Consts.cycle_loss_lambda
        self.identity_loss_lambda = Consts.identity_loss_lambda

        self.number_of_frames = Consts.number_of_frames
        self.batch_size = Consts.mini_batch_size

        self.validator = Validator(A_cache_dir, B_cache_dir)

        self.A2B_gen = GeneratorCycleGan2()
        self.B2A_gen = GeneratorCycleGan2()
        self.A_disc = DiscriminatorCycleGan2()
        self.B_disc = DiscriminatorCycleGan2()

    def configure_optimizers(self):
        discriminators_params = list(self.A_disc.parameters()) + list(self.B_disc.parameters())
        generators_params = list(self.A2B_gen.parameters()) + list(self.B2A_gen.parameters())

        discriminators_optimizer = torch.optim.Adam(
            discriminators_params,
            lr=self.disc_lr,
            betas=Consts.adam_optimizer_betas
        )
        generators_optimizer = torch.optim.Adam(
            generators_params,
            lr=self.gen_lr,
            betas=Consts.adam_optimizer_betas
        )

        # TODO: schedulers for decaying lr
        return [discriminators_optimizer, generators_optimizer], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A, real_B = batch

        if optimizer_idx == 0:
            d_loss = self._discriminators_step(real_A, real_B)
            return {"loss": d_loss}

        if optimizer_idx == 1:
            g_loss = self._generators_step(real_A, real_B)
            return {"loss": g_loss}

    def forward(self, A_source, B_source):
        return self.A2B_gen(A_source), self.B2A_gen(B_source)

    def train_dataloader(self):
        dataset = PreprocessedDataset(
            A_dataset=self.hparams.A_dataset,
            B_dataset=self.hparams.B_dataset,
            number_of_frames=self.number_of_frames
        )

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            # num_workers=12 #  if set, then: `The paging file is too small for this operation to complete.`
        )
        return dataloader

    def val_dataloader(self):
        A_dir = self.hparams.A_validation_source_dir
        B_dir = self.hparams.B_validation_source_dir

        dataset = LightningValidationDataset(A_dir, B_dir, self.validator)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False
        )
        return dataloader

    def validation_step(self, batch, batch_idx):
        # that's the uglies thing in this whole repo, please don't judge me here
        A, B = batch
        (A_signal, (A_f0, A_ap)), A_file_name = A
        (B_signal, (B_f0, B_ap)), B_file_name = B

        A_output_path = os.path.join(self.hparams.A2B_validation_output_dir, A_file_name[0])
        B_output_path = os.path.join(self.hparams.B2A_validation_output_dir, B_file_name[0])

        B_generated, A_generated = self(A_signal[0], B_signal[0])

        d_fake_B = self.B_disc(B_generated)
        d_fake_A = self.A_disc(A_generated)
        B2A_adv_loss = self._adversarial_loss(d_fake_A, torch.ones_like(d_fake_A))
        A2B_adv_loss = self._adversarial_loss(d_fake_B, torch.ones_like(d_fake_B))
        gen_adversarial_loss = B2A_adv_loss + A2B_adv_loss

        self.validator.denormalize_and_save(signal=B_generated, ap=A_ap.cpu().numpy()[0], f0=A_f0.cpu().numpy()[0], file_path=A_output_path, is_A=False)
        self.validator.denormalize_and_save(signal=A_generated, ap=B_ap.cpu().numpy()[0], f0=B_f0.cpu().numpy()[0], file_path=B_output_path, is_A=True)

        return {"val_loss": gen_adversarial_loss}

    def on_train_epoch_end(self) -> None:
        print(self.current_epoch)

    def on_validation_start(self) -> None:
        print("test")

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(avg_loss)

    @staticmethod
    def _adversarial_loss(output, expected):
        return torch.mean((expected - output) ** 2)

    @staticmethod
    def _cycle_loss(real, cycle):
        return torch.mean(torch.abs(real - cycle))

    @staticmethod
    def _identity_loss(real, identity):
        return torch.mean(torch.abs(real - identity))

    def _discriminators_step(self, real_A, real_B):
        fake_B, fake_A = self(real_A, real_B)

        d_fake_B = self.B_disc(fake_B)
        d_fake_A = self.A_disc(fake_A)

        d_real_A = self.A_disc(real_A)
        d_real_B = self.B_disc(real_B)

        d_fake_adv_loss_A = self._adversarial_loss(d_fake_A, torch.zeros_like(d_fake_A))
        d_real_adv_loss_A = self._adversarial_loss(d_real_A, torch.ones_like(d_real_A))
        d_adv_loss_A = d_fake_adv_loss_A + d_real_adv_loss_A

        d_fake_adv_loss_B = self._adversarial_loss(d_fake_B, torch.zeros_like(d_fake_B))
        d_real_adv_loss_B = self._adversarial_loss(d_real_B, torch.ones_like(d_real_B))
        d_adv_loss_B = d_fake_adv_loss_B + d_real_adv_loss_B

        d_loss = (d_adv_loss_A + d_adv_loss_B) / 2.0
        return d_loss

    def _generators_step(self, real_A, real_B):
        fake_B, fake_A = self(real_A, real_B)

        d_fake_A = self.A_disc(fake_A)
        d_fake_B = self.B_disc(fake_B)
        B2A_adv_loss = self._adversarial_loss(d_fake_A, torch.ones_like(d_fake_A))
        A2B_adv_loss = self._adversarial_loss(d_fake_B, torch.ones_like(d_fake_B))
        gen_adversarial_loss = B2A_adv_loss + A2B_adv_loss

        cycle_A = self.B2A_gen(fake_B)
        cycle_B = self.A2B_gen(fake_A)
        cycle_loss_A = self._cycle_loss(real_A, cycle_A)
        cycle_loss_B = self._cycle_loss(real_B, cycle_B)
        cycle_loss = self.cycle_loss_lambda * (cycle_loss_A + cycle_loss_B)

        if self.identity_loss_lambda == 0:  # do not compute for nothing
            identity_loss = torch.zeros_like(cycle_loss)
        else:
            identity_A = self.B2A_gen(real_A)
            identity_B = self.A2B_gen(real_B)
            identity_loss_A = self._identity_loss(real_A, identity_A)
            identity_loss_B = self._identity_loss(real_B, identity_B)
            identity_loss = self.identity_loss_lambda * (identity_loss_A + identity_loss_B)

        g_loss = gen_adversarial_loss + cycle_loss + identity_loss
        return g_loss


if __name__ == '__main__':
    A_dir, B_dir = Consts.female_to_male
    print(f"FROM: {A_dir} TO: {B_dir}")

    download_destination = Consts.vc16_data_directory_path
    training_data_dir = Consts.vc16_training_directory_path
    validation_data_dir = Consts.vc16_validation_directory_path
    A_validation_source_dir = os.path.join(validation_data_dir, A_dir)
    B_validation_source_dir = os.path.join(validation_data_dir, B_dir)

    A_dataset = FilesOperator.load_pickle_file(Consts.A_preprocessed_dataset_file_path)
    B_dataset = FilesOperator.load_pickle_file(Consts.B_preprocessed_dataset_file_path)
    trainer = pl.Trainer(fast_dev_run=False,
                         gpus=1,
                         check_val_every_n_epoch=2,#Consts.dump_validation_file_epoch_frequency,
                         enable_progress_bar=False
                         )
    cycle_gan = LightningCycleGan(
        A_dataset=A_dataset,
        B_dataset=B_dataset,
        A_validation_source_dir=A_validation_source_dir,
        B_validation_source_dir=B_validation_source_dir,
        A2B_validation_output_dir=Consts.A2B_validation_output_directory_path,
        B2A_validation_output_dir=Consts.B2A_validation_output_directory_path,
        A_cache_dir=Consts.A_cache_directory_path,
        B_cache_dir=Consts.B_cache_directory_path
    )
    with np.errstate(divide='ignore'):  # np.log 'throws' warning
        trainer.fit(cycle_gan)
