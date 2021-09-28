import os
import torch
from torch.utils.data import DataLoader

from consts import Consts
from src.utils.utils import Utils

from src.data_processing.dataset import PreprocessedDataset
from src.model.generator import Generator
from src.model.discriminator import Discriminator
from src.data_processing.validator import Validator


class CycleGanTraining:
    def __init__(self,  # only dataset parameters are given explicitly in constructor
                 A_data_file,
                 B_data_file,
                 number_of_frames,
                 batch_size,
                 A_validation_dir,
                 B_validation_dir,
                 A_output_dir,
                 B_output_dir,
                 A_cache_dir,
                 B_cache_dir):

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

        self.gen_lr = Consts.generator_lr
        self.disc_lr = Consts.discriminator_lr

        self.gen_lr_decay = Consts.generator_lr_decay
        self.disc_lr_decay = Consts.discriminator_lr_decay

        self.gen_optimizer = \
            torch.optim.Adam(gen_params, lr=self.gen_lr, betas=Consts.adam_optimizer_betas)
        self.disc_optimizer = \
            torch.optim.Adam(disc_params, lr=self.disc_lr, betas=Consts.adam_optimizer_betas)

        self.gen_loss_store = []
        self.disc_loss_store = []

        # ------------------------------ #
        #  validation                    #
        # ------------------------------ #
        self.validator = CycleGanTraining._prepare_validator(A_cache_dir, B_cache_dir)
        self.A_validation_dir = A_validation_dir
        self.B_validation_dir = B_validation_dir
        self.A_output_dir = A_output_dir
        self.B_output_dir = B_output_dir
        self.dump_validation_file_epoch_frequency = Consts.dump_validation_file_epoch_frequency
        self.print_losses_iteration_frequency = Consts.print_losses_iteration_frequency

    def train(self):
        for epoch_num in range(self.number_of_epochs):
            # print(f"Epoch {epoch_num + 1}")
            self._train_single_epoch(epoch_num)
            if (epoch_num + 1) % self.dump_validation_file_epoch_frequency == 0:
                self._validate(epoch_num + 1)
        print("Finished training")

    def _train_single_epoch(self, epoch_num):
        for i, (real_A, real_B) in enumerate(self.dataloader):
            iteration = (self.number_of_samples_in_dataset // self.batch_size) * epoch_num + i
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            # ------------------------------ #
            #  modify parameters             #
            # ------------------------------ #
            if iteration > self.zero_identity_lambda_loss_after:
                self.identity_loss_lambda = 0
            if iteration > self.start_decay_after:
                self._adjust_generator_lr()
                self._adjust_discriminator_lr()

            # ------------------------------ #
            #  GENERATORS                    #
            # ------------------------------ #
            fake_B = self.A2B_gen(real_A)
            cycle_A = self.B2A_gen(fake_B)

            fake_A = self.B2A_gen(real_B)
            cycle_B = self.B2A_gen(fake_A)

            identity_A = self.B2A_gen(real_A)
            identity_B = self.A2B_gen(real_B)

            d_fake_A = self.A_disc(fake_A)
            d_fake_B = self.B_disc(fake_B)

            # ------------------------------ #
            #  count generator loss          #
            # ------------------------------ #
            cycle_loss = torch.mean(torch.abs(real_A - cycle_A)) + \
                         torch.mean(torch.abs(real_B - cycle_B))

            identity_loss = torch.mean(torch.abs(real_A - identity_A)) + \
                            torch.mean(torch.abs(real_B - identity_B))

            A2B_gen_loss = torch.mean((1 - d_fake_B) ** 2)
            B2A_gen_loss = torch.mean((1 - d_fake_A) ** 2)

            generator_loss = A2B_gen_loss + \
                             B2A_gen_loss + \
                             self.cycle_loss_lambda * cycle_loss + \
                             self.identity_loss_lambda * identity_loss
            self.gen_loss_store.append(generator_loss.item())

            self._reset_grad()
            generator_loss.backward()
            self.gen_optimizer.step()

            # ------------------------------ #
            #  DISCRIMINATORS                #
            # ------------------------------ #
            d_real_A = self.A_disc(real_A)
            d_real_B = self.B_disc(real_B)

            fake_A = self.B2A_gen(real_B)
            d_fake_A = self.A_disc(fake_A)

            fake_B = self.A2B_gen(real_A)
            d_fake_B = self.B_disc(fake_B)

            # ------------------------------ #
            #  count discriminator loss      #
            # ------------------------------ #
            d_loss_A = CycleGanTraining._count_discriminator_loss(d_real_A, d_fake_A)
            d_loss_B = CycleGanTraining._count_discriminator_loss(d_real_B, d_fake_B)

            d_loss = (d_loss_A + d_loss_B) / 2.0
            self.disc_loss_store.append(d_loss)

            self._reset_grad()
            d_loss.backward()
            self.disc_optimizer.step()

            # ------------------------------ #
            #  printing                      #
            # ------------------------------ #
            if (iteration + 1) % self.print_losses_iteration_frequency == 0:
                CycleGanTraining._print_losses(iteration=iteration,
                                               generator_loss=generator_loss,
                                               discriminator_loss=d_loss,
                                               cycle_loss=cycle_loss,
                                               identity_loss=identity_loss)

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

    @staticmethod
    def _prepare_validator(A_cache_dir, B_cache_dir):
        validator = Validator(
            A_cache_dir=A_cache_dir,
            B_cache_dir=B_cache_dir
        )

        return validator

    def _adjust_generator_lr(self):
        self.gen_lr = max(0., self.gen_lr - self.gen_lr_decay)
        for param_groups in self.gen_optimizer.param_groups:
            param_groups['lr'] = self.gen_lr

    def _adjust_discriminator_lr(self):
        self.disc_lr = max(0., self.disc_lr - self.disc_lr_decay)
        for param_groups in self.disc_optimizer.param_groups:
            param_groups['lr'] = self.disc_lr

    def _reset_grad(self):
        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()

    @staticmethod
    def _count_discriminator_loss(d_real, d_fake):
        d_loss_real = torch.mean((1 - d_real) ** 2)
        d_loss_fake = torch.mean(d_fake ** 2)

        return (d_loss_real + d_loss_fake) / 2.0

    @staticmethod
    def _print_losses(iteration, generator_loss, discriminator_loss, cycle_loss, identity_loss):
        losses_str = f"{iteration + 1}: \n" + \
                     f"\tGenerator-loss:     {generator_loss.item():.4f}\n" + \
                     f"\tDiscriminator-loss: {discriminator_loss.item():.4f}\n" + \
                     f"\tCycle-loss:         {cycle_loss.item():.4f}\n" + \
                     f"\tIdentity-loss:      {identity_loss.item():.4f}\n"
        losses_str = losses_str.replace("\n", "")
        print(losses_str)

    def _validate(self, epoch):
        self._validate_single_generator(epoch=epoch,
                                        generator=self.A2B_gen,
                                        validation_directory=self.A_validation_dir,
                                        output_dir=self.A_output_dir,
                                        is_A=True)

        self._validate_single_generator(epoch=epoch,
                                        generator=self.B2A_gen,
                                        validation_directory=self.B_validation_dir,
                                        output_dir=self.B_output_dir,
                                        is_A=False)

    def _validate_single_generator(self, epoch, generator, validation_directory, output_dir, is_A):
        epoch_output_dir = os.path.join(output_dir, str(epoch))
        os.mkdir(epoch_output_dir)
        for file in os.listdir(validation_directory):
            file_path = os.path.join(validation_directory, file)
            output_file_path = os.path.join(epoch_output_dir, file)

            input_signal, (f0, ap) = self.validator.load_and_normalize(file_path=file_path, is_A=is_A)
            signal_tensor = torch.from_numpy(input_signal)
            ready_signal = signal_tensor.to(self.device).float()
            generated = generator(ready_signal)
            detached = generated.cpu().detach()

            self.validator.denormalize_and_save(signal=detached,
                                                ap=ap,
                                                f0=f0,
                                                file_path=output_file_path,
                                                is_A=is_A)
