import os
import torch
from typing import Optional
from torch.utils.data import DataLoader

from consts import Consts
from src.utils.files_operator import FilesOperator
from src.utils.utils import Utils

from src.data_processing.dataset import PreprocessedDataset
from src.data_processing.validator import Validator

from src.model.cycle_gan_vc.generator import Generator
from src.model.cycle_gan_vc.discriminator import Discriminator

from src.model.cycle_gan_vc2.generator import GeneratorCycleGan2
from src.model.cycle_gan_vc2.discriminator import DiscriminatorCycleGan2


class CycleGanTraining:
    # only directories are given explicitly in constructor, rest training parameters are given in the `const.py` file
    def __init__(self,
                 A_data_file,
                 B_data_file,
                 A_validation_source_dir,
                 B_validation_source_dir,
                 A2B_validation_output_dir,
                 B2A_validation_output_dir,
                 A_cache_dir,
                 B_cache_dir,
                 save_models_dir: str,
                 load_models_dir: Optional[str]):

        # ------------------------------ #
        #  hyper parameters              #
        # ------------------------------ #
        self.number_of_epochs = Consts.number_of_epochs
        self.batch_size = Consts.mini_batch_size
        self.cycle_loss_lambda = Consts.cycle_loss_lambda
        self.identity_loss_lambda = Consts.identity_loss_lambda
        self.zero_identity_lambda_loss_after = Consts.zero_identity_loss_lambda_after
        self.start_decay_after = Consts.start_decay_after
        self.device = Utils.get_device()
        self.MSE_fn = torch.nn.MSELoss()
        self.L1L_fn = torch.nn.L1Loss()

        # ------------------------------ #
        #  dataloader                    #
        # ------------------------------ #
        self.number_of_frames = Consts.number_of_frames
        self.dataset = CycleGanTraining._prepare_dataset(A_data_file, B_data_file, self.number_of_frames)
        self.dataset.prepare_and_shuffle()
        self.dataloader = CycleGanTraining._prepare_dataloader(self.dataset, self.batch_size)
        self.number_of_samples_in_dataset = len(self.dataset)

        # ------------------------------ #
        #  generators and discriminators #
        # ------------------------------ #
        # self.A2B_gen = Generator().to(self.device)
        # self.B2A_gen = Generator().to(self.device)
        # self.A_disc = Discriminator().to(self.device)
        # self.B_disc = Discriminator().to(self.device)
        self.A2B_gen = GeneratorCycleGan2().to(self.device)
        self.B2A_gen = GeneratorCycleGan2().to(self.device)
        self.A_disc = DiscriminatorCycleGan2().to(self.device)
        self.B_disc = DiscriminatorCycleGan2().to(self.device)

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
        self.A_validation_source_dir = A_validation_source_dir
        self.B_validation_source_dir = B_validation_source_dir
        self.A2B_validation_output_dir = A2B_validation_output_dir
        self.B2A_validation_output_dir = B2A_validation_output_dir
        self.dump_validation_file_epoch_frequency = Consts.dump_validation_file_epoch_frequency
        self.print_losses_iteration_frequency = Consts.print_losses_iteration_frequency
        self.log_file_name = Consts.log_file_path
        self._log_message('\n\n---------\nNEXT RUN\n---------\n')

        # ------------------------------ #
        #  model storage                 #
        # ------------------------------ #
        self.models_saving_epoch_frequency = Consts.models_saving_epoch_frequency
        self.save_models_directory = save_models_dir
        self.load_models_directory = load_models_dir
        if self.load_models_directory:
            self._load_models()

    def train(self):
        for epoch_num in range(self.number_of_epochs):
            # print(f"Epoch {epoch_num + 1}")
            self.dataset.prepare_and_shuffle()
            self.dataloader = CycleGanTraining._prepare_dataloader(self.dataset, self.batch_size)
            # self._train_single_epoch(epoch_num)
            self._train_single_epoch_overhauled(epoch_num)

            if (epoch_num + 1) % self.dump_validation_file_epoch_frequency == 0:
                # print("Dumping validation files... ", end='')
                self._validate(epoch_num + 1)
                # print("Done")

            if (epoch_num + 1) % self.models_saving_epoch_frequency == 0:
                # print("Checkpoint... ", end='')
                self._checkpoint()
                # print("Done")

        print("Finished training")

    def _train_single_epoch(self, epoch_num):
        for i, (real_A, real_B) in enumerate(self.dataloader):
            iteration = (self.number_of_samples_in_dataset // self.batch_size) * epoch_num + i
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            # ------------------------------ #
            #  modify parameters             #
            # ------------------------------ #
            self._adjust_params(iteration)

            # ------------------------------ #
            #  training discriminators       #
            # ------------------------------ #
            fake_A = self.B2A_gen(real_B)
            fake_B = self.A2B_gen(real_A)

            d_fake_A = self.A_disc(fake_A.detach())
            d_fake_B = self.B_disc(fake_B.detach())

            d_real_A = self.A_disc(real_A)
            d_real_B = self.B_disc(real_B)

            d_fake_loss_A = self._adversarial_loss_fn(d_fake_A, torch.zeros_like(d_fake_A))
            d_real_loss_A = self._adversarial_loss_fn(d_real_A, torch.ones_like(d_real_A))
            d_loss_A = d_fake_loss_A + d_real_loss_A

            d_fake_loss_B = self._adversarial_loss_fn(d_fake_B, torch.zeros_like(d_fake_B))
            d_real_loss_B = self._adversarial_loss_fn(d_real_B, torch.ones_like(d_real_B))
            d_loss_B = d_fake_loss_B + d_real_loss_B

            d_loss = (d_loss_A + d_loss_B) / 2.0  # todo: what is dividing by two for?

            self.disc_optimizer.zero_grad()
            d_loss.backward()
            self.disc_optimizer.step()

            # ------------------------------ #
            #  training generators           #
            # ------------------------------ #
            d_fake_A = self.A_disc(fake_A)
            d_fake_B = self.B_disc(fake_B)
            adv_loss_A = self._adversarial_loss_fn(d_fake_A, torch.ones_like(d_fake_A))
            adv_loss_B = self._adversarial_loss_fn(d_fake_B, torch.ones_like(d_fake_B))
            adversarial_loss = adv_loss_A + adv_loss_B

            cycle_A = self.B2A_gen(fake_B)
            cycle_B = self.A2B_gen(fake_A)
            cycle_loss_A = self._cycle_loss_fn(real_A, cycle_A)
            cycle_loss_B = self._cycle_loss_fn(real_B, cycle_B)
            cycle_loss = self.cycle_loss_lambda * cycle_loss_A + self.cycle_loss_lambda * cycle_loss_B

            identity_A = self.B2A_gen(real_A)
            identity_B = self.A2B_gen(real_B)
            identity_loss_A = self._identity_loss_fn(real_A, identity_A)
            identity_loss_B = self._identity_loss_fn(real_B, identity_B)
            identity_loss = self.identity_loss_lambda * identity_loss_A + self.identity_loss_lambda * identity_loss_B

            g_loss = adversarial_loss + cycle_loss + identity_loss

            self.gen_optimizer.zero_grad()
            g_loss.backward()
            self.gen_optimizer.step()

            # ------------------------------ #
            #  analytics                     #
            # ------------------------------ #
            self._analytics(
                iteration=iteration,
                g_loss=g_loss,
                d_loss=d_loss,
                cycle_loss=cycle_loss,
                identity_loss=identity_loss
            )

    def _train_single_epoch_overhauled(self, epoch_num):
        for i, (real_A, real_B) in enumerate(self.dataloader):
            iteration = (self.number_of_samples_in_dataset // self.batch_size) * epoch_num + i
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)

            # ------------------------------ #
            #  modify parameters             #
            # ------------------------------ #
            self._adjust_params(iteration)

            # ------------------------------ #
            #  GENERATOR                     #
            # ------------------------------ #
            fake_B = self.A2B_gen(real_A)
            cycle_A = self.B2A_gen(fake_B)

            fake_A = self.B2A_gen(real_B)
            cycle_B = self.A2B_gen(fake_A)

            identity_A = self.B2A_gen(real_A)
            identity_B = self.A2B_gen(real_B)

            d_fake_A = self.A_disc(fake_A)
            d_fake_B = self.B_disc(fake_B)

            cycle_loss = torch.mean(torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))
            identity_loss = torch.mean(torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))
            generator_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
            generator_loss_B2A = torch.mean((1 - d_fake_A) ** 2)
            g_loss = generator_loss_A2B + generator_loss_B2A + \
                     self.cycle_loss_lambda * cycle_loss + \
                     self.identity_loss_lambda * identity_loss

            self._reset_grad()
            g_loss.backward()
            self.gen_optimizer.step()

            # ------------------------------ #
            #  DISCRIMINATOR                 #
            # ------------------------------ #
            d_real_A = self.A_disc(real_A)
            d_real_B = self.B_disc(real_B)

            generated_A = self.B2A_gen(real_B)
            d_fake_A = self.A_disc(generated_A)

            generated_B = self.A2B_gen(real_A)
            d_fake_B = self.B_disc(generated_B)

            # Loss Functions
            d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
            d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
            d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

            d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
            d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
            d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

            # TODO: try with d_cycle
            # cycled_B = self.generator_A2B(generated_A)
            # d_cycled_B = self.discriminator_B(cycled_B)
            # cycled_A = self.generator_B2A(generated_B)
            # d_cycled_A = self.discriminator_A(cycled_A)
            # d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2)
            # d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)
            # d_loss_A_2nd = (d_loss_A_real + d_loss_A_cycled) / 2.0
            # d_loss_B_2nd = (d_loss_B_real + d_loss_B_cycled) / 2.0

            d_loss = (d_loss_A + d_loss_B) / 2.0  # TODO+ (d_loss_A_2nd + d_loss_B_2nd) / 2.0

            self._reset_grad()
            d_loss.backward()
            self.disc_optimizer.step()

            # ------------------------------ #
            #  analytics                     #
            # ------------------------------ #
            self._analytics(
                iteration=iteration,
                g_loss=g_loss,
                d_loss=d_loss,
                cycle_loss=cycle_loss,
                identity_loss=identity_loss,
                A2B_loss=generator_loss_A2B,
                B2A_loss=generator_loss_B2A,
                d_A_loss=d_loss_A,
                d_B_loss=d_loss_B
            )

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
            shuffle=True,
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

    def _adjust_lr(self):
        self._adjust_discriminator_lr()
        self._adjust_generator_lr()

    def _adjust_generator_lr(self):
        self.gen_lr = max(0., self.gen_lr - self.gen_lr_decay)
        for param_groups in self.gen_optimizer.param_groups:
            param_groups['lr'] = self.gen_lr

    def _adjust_discriminator_lr(self):
        self.disc_lr = max(0., self.disc_lr - self.disc_lr_decay)
        for param_groups in self.disc_optimizer.param_groups:
            param_groups['lr'] = self.disc_lr

    def _adversarial_loss_fn(self, x, y):
        return self.MSE_fn(x, y)

    def _cycle_loss_fn(self, x, y):
        return self.L1L_fn(x, y)

    def _identity_loss_fn(self, x, y):
        return self.L1L_fn(x, y)

    def _validate(self, epoch):
        self._validate_single_generator(epoch=epoch,
                                        generator=self.A2B_gen,
                                        validation_directory=self.A_validation_source_dir,
                                        output_dir=self.A2B_validation_output_dir,
                                        is_A2B=True)

        self._validate_single_generator(epoch=epoch,
                                        generator=self.B2A_gen,
                                        validation_directory=self.B_validation_source_dir,
                                        output_dir=self.B2A_validation_output_dir,
                                        is_A2B=False)

    def _validate_single_generator(self, epoch, generator, validation_directory, output_dir, is_A2B):
        epoch_output_dir = os.path.join(output_dir, str(epoch))
        os.mkdir(epoch_output_dir)
        for file in os.listdir(validation_directory):
            file_path = os.path.join(validation_directory, file)
            output_file_path = os.path.join(epoch_output_dir, file)

            with torch.no_grad():
                input_signal, (f0, ap) = self.validator.load_and_normalize(file_path=file_path, is_A=is_A2B)

                signal_tensor = torch.from_numpy(input_signal)
                device_input = signal_tensor.to(self.device).float()
                device_generated = generator(device_input)
                cpu_generated = device_generated.cpu()

                self.validator.denormalize_and_save(signal=cpu_generated,
                                                    ap=ap,
                                                    f0=f0,
                                                    file_path=output_file_path,
                                                    is_A=not is_A2B)  # negation, because now we are in the "opposite" domain

    def _checkpoint(self):
        save_dir = self.save_models_directory
        self._save_models(save_dir)
        self._save_models_losses(save_dir)

    def _load_models(self):
        load_dir = self.load_models_directory
        self.A2B_gen.load_state_dict(FilesOperator.load_model(load_dir, Consts.A2B_generator_file_name))
        self.B2A_gen.load_state_dict(FilesOperator.load_model(load_dir, Consts.B2A_generator_file_name))
        self.A_disc.load_state_dict(FilesOperator.load_model(load_dir, Consts.A_discriminator_file_name))
        self.B_disc.load_state_dict(FilesOperator.load_model(load_dir, Consts.B_discriminator_file_name))

    def _save_models(self, save_dir):
        FilesOperator.save_model(self.A2B_gen, save_dir, Consts.A2B_generator_file_name)
        FilesOperator.save_model(self.B2A_gen, save_dir, Consts.B2A_generator_file_name)
        FilesOperator.save_model(self.A_disc, save_dir, Consts.A_discriminator_file_name)
        FilesOperator.save_model(self.B_disc, save_dir, Consts.B_discriminator_file_name)

    def _save_models_losses(self, save_dir):
        FilesOperator.save_list(self.gen_loss_store, save_dir, Consts.generator_loss_storage_file)
        FilesOperator.save_list(self.disc_loss_store, save_dir, Consts.discriminator_loss_storage_file)

    def _adjust_params(self, iteration):
        if iteration > self.zero_identity_lambda_loss_after:
            self.identity_loss_lambda = 0
        if iteration > self.start_decay_after:
            self._adjust_lr()

    def _analytics(self, iteration, g_loss, d_loss, cycle_loss, identity_loss,
                   A2B_loss=torch.tensor(0), B2A_loss=torch.tensor(0),
                   d_A_loss=torch.tensor(0), d_B_loss=torch.tensor(0)):
        if iteration % self.print_losses_iteration_frequency == 0:
            self._print_losses_and_log(iteration=iteration,
                                       generator_loss=g_loss,
                                       discriminator_loss=d_loss,
                                       cycle_loss=cycle_loss,
                                       identity_loss=identity_loss,
                                       A2B_loss=A2B_loss,
                                       B2A_loss=B2A_loss,
                                       d_A_loss=d_A_loss,
                                       d_B_loss=d_B_loss)
        if iteration % (10 * self.print_losses_iteration_frequency) == 0:
            self._print_params()
        self.gen_loss_store.append(g_loss.cpu().detach().item())
        self.disc_loss_store.append(d_loss.cpu().detach().item())

    def _print_losses_and_log(self, iteration, generator_loss, discriminator_loss, cycle_loss, identity_loss,
                              A2B_loss, B2A_loss, d_A_loss, d_B_loss):
        losses_str = f"{iteration:>5}: \n" + \
                     f"\tG_loss: {generator_loss.item():.4f}\n" + \
                     f"\tD_loss: {discriminator_loss.item():.4f}\n" + \
                     f"\tCycle_loss: {cycle_loss.item():.4f}\n" + \
                     f"\tIdentity_loss:{identity_loss.item():.4f}\n" + \
                     f"\tA2B_loss: {A2B_loss.item():.4f}\n" + \
                     f"\tB2A_loss: {B2A_loss.item():.4f}\n" + \
                     f"\td_A_loss: {d_A_loss.item():.4f}\n" + \
                     f"\td_B_loss: {d_B_loss.item():.4f}\n"
        losses_str = losses_str.replace("\n", "")
        print(losses_str)
        self._log_message(losses_str + '\n')

    def _print_params(self):
        params_str = f"gen_lr: {self.gen_lr}, disc_lr:{self.disc_lr}"
        print(params_str)

    def _reset_grad(self):  # just for the overhauled version
        self.gen_optimizer.zero_grad()
        self.disc_optimizer.zero_grad()

    def _log_message(self, msg):
        f = open(self.log_file_name, "a")
        f.write(msg)
        f.close()