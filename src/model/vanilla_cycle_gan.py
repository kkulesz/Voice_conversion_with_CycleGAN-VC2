import os
import torch
import wandb
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


class VanillaCycleGan:
    def __init__(self,
                 A_dataset,
                 B_dataset,
                 A_validation_source_dir,
                 B_validation_source_dir,
                 A2B_validation_output_dir,
                 B2A_validation_output_dir,
                 A_cache_dir,
                 B_cache_dir,
                 save_models_dir: str,
                 load_models_dir: Optional[str],
                 start_from_epoch_number: int):

        # ------------------------------ #
        #  hyper parameters              #
        # ------------------------------ #
        self.start_from_epoch_number = start_from_epoch_number
        self.number_of_epochs = Consts.number_of_epochs
        self.batch_size = Consts.mini_batch_size
        self.cycle_loss_lambda = Consts.cycle_loss_lambda
        self.identity_loss_lambda = Consts.identity_loss_lambda
        self.zero_identity_lambda_loss_after = Consts.zero_identity_loss_lambda_after
        self.start_decay_after = Consts.start_decay_after
        self.device = Utils.get_device()

        # ------------------------------ #
        #  dataloader                    #
        # ------------------------------ #
        self.number_of_frames = Consts.number_of_frames
        self.dataset = self._prepare_dataset(A_dataset, B_dataset, self.number_of_frames)
        self.dataloader = self._prepare_dataloader(self.dataset, self.batch_size)
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

        if self.start_from_epoch_number > Consts.start_decay_after:
            epochs_over = Consts.start_decay_after - self.start_from_epoch_number
            self.gen_lr = max(0, self.gen_lr - epochs_over * Consts.generator_lr_decay)
            self.disc_lr = max(0, self.disc_lr - epochs_over * Consts.discriminator_lr_decay)

        self.gen_optimizer = \
            torch.optim.Adam(gen_params, lr=self.gen_lr, betas=Consts.adam_optimizer_betas)
        self.disc_optimizer = \
            torch.optim.Adam(disc_params, lr=self.disc_lr, betas=Consts.adam_optimizer_betas)

        self.gen_loss_store = []
        self.disc_loss_store = []

        # ------------------------------ #
        #  validation                    #
        # ------------------------------ #
        self.validator = self._prepare_validator(A_cache_dir, B_cache_dir)
        self.A_validation_source_dir = A_validation_source_dir
        self.B_validation_source_dir = B_validation_source_dir
        self.A2B_validation_output_dir = A2B_validation_output_dir
        self.B2A_validation_output_dir = B2A_validation_output_dir
        self.dump_validation_file_epoch_frequency = Consts.dump_validation_file_epoch_frequency
        self.print_losses_iteration_frequency = Consts.print_losses_iteration_frequency
        self.log_file_name = Consts.log_file_path
        self._log_message('\n\n---------\nNEXT RUN\n---------\n')
        # ------------------------------ #
        #  weightAndBiases boilerplate   #
        # ------------------------------ #
        wandb.watch(models=self.B2A_gen,
                    criterion=None,
                    log="all",
                    log_freq=10)
        # wandb.watch(models=self.A_disc,
        #             criterion=None,
        #             log="all",
        #             log_freq=10,
        #             log_graph=True
        #             )
        self.val_A, self.val_B = next(iter(self.dataloader))

        # ------------------------------ #
        #  model storage                 #
        # ------------------------------ #
        self.models_saving_epoch_frequency = Consts.models_saving_epoch_frequency
        self.save_models_directory = save_models_dir
        self.load_models_directory = load_models_dir
        if self.load_models_directory:
            self._load_models()

    def train(self):
        for epoch_num in range(self.start_from_epoch_number, self.number_of_epochs):
            # print(f"Epoch {epoch_num + 1}")
            self._train_single_epoch(epoch_num)

            if (epoch_num + 1) % self.dump_validation_file_epoch_frequency == 0:
                print(f"Dumping validation files after epoch {epoch_num + 1}... ", end='')
                self._validate(epoch_num + 1)
                print("Done")

            if (epoch_num + 1) % self.models_saving_epoch_frequency == 0:
                print(f"Checkpoint after epoch {epoch_num + 1}... ", end='')
                self._checkpoint()
                print("Done")
            self._checkpoint()

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

            d_fake_adv_loss_A = self._adversarial_loss(d_fake_A, torch.zeros_like(d_fake_A))
            d_real_adv_loss_A = self._adversarial_loss(d_real_A, torch.ones_like(d_real_A))
            d_adv_loss_A = d_fake_adv_loss_A + d_real_adv_loss_A

            d_fake_adv_loss_B = self._adversarial_loss(d_fake_B, torch.zeros_like(d_fake_B))
            d_real_adv_loss_B = self._adversarial_loss(d_real_B, torch.ones_like(d_real_B))
            d_adv_loss_B = d_fake_adv_loss_B + d_real_adv_loss_B

            d_loss = (d_adv_loss_A + d_adv_loss_B) / 2.0

            self.disc_optimizer.zero_grad()
            d_loss.backward()
            self.disc_optimizer.step()

            # ------------------------------ #
            #  training generators           #
            # ------------------------------ #
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
                identity_loss=identity_loss,
                A2B_loss=A2B_adv_loss,
                B2A_loss=B2A_adv_loss,
                d_A_loss=d_adv_loss_A,
                d_B_loss=d_adv_loss_B
            )

    @staticmethod
    def _prepare_dataset(A_dataset, B_dataset, number_of_frames):
        return PreprocessedDataset(
            A_dataset=A_dataset,
            B_dataset=B_dataset,
            number_of_frames=number_of_frames
        )

    @staticmethod
    def _prepare_dataloader(dataset, batch_size):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

    @staticmethod
    def _prepare_validator(A_cache_dir, B_cache_dir):
        return Validator(
            A_cache_dir=A_cache_dir,
            B_cache_dir=B_cache_dir
        )

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

    @staticmethod
    def _adversarial_loss(output, expected):
        return torch.mean((expected - output) ** 2)

    @staticmethod
    def _cycle_loss(real, cycle):
        return torch.mean(torch.abs(real - cycle))

    @staticmethod
    def _identity_loss(real, identity):
        return torch.mean(torch.abs(real - identity))

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

                device_input = input_signal.to(self.device).float()
                device_generated = generator(device_input)

                self.validator.denormalize_and_save(signal=device_generated,
                                                    ap=ap,
                                                    f0=f0,
                                                    file_path=output_file_path,
                                                    is_A=not is_A2B)  # negation, because now we are in the "opposite" domain

    def _checkpoint(self):
        save_dir = self.save_models_directory
        self._save_models(save_dir)
        self._save_models_losses(save_dir)

        # COMMENTED OUT
        # file = "B2A.onnx"
        # save_path = os.path.join(wandb.run.dir, file)  # `wandb` is bugged, this is workaround to avoid creating symbolic link
        # torch.onnx.export(
        #     self.B2A_gen,
        #     self.val_B.to(self.device),
        #     save_path)
        # wandb.save(file)

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

        wandb.log(
            {'generator_loss': generator_loss,
             'discriminator_loss': discriminator_loss,
             'cycle_loss': cycle_loss,
             'identity_loss': identity_loss}, step=iteration
        )

    def _print_params(self):
        params_str = f"gen_lr: {self.gen_lr}, disc_lr:{self.disc_lr}, id_loss_lambda: {self.identity_loss_lambda}"
        print(params_str)

    def _log_message(self, msg):
        f = open(self.log_file_name, "a")
        f.write(msg)
        f.close()
