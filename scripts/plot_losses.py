import os
import pandas as pd
import matplotlib.pyplot as plt

from consts import Consts


if __name__ == '__main__':
    directory_path = Consts.models_storage_directory_path
    # directory_path = 'E:\\STUDIA\inzynierka\\2_moje_przygotowania\\4.trenowanie\\2.SM1-TM1\\models_storage'
    discriminators_losses_file_path = os.path.join(directory_path, Consts.discriminator_loss_storage_file)
    generators_losses_file_path = os.path.join(directory_path, Consts.generator_loss_storage_file)

    plot_after = int(5 * 1e+3)
    plot_before = int(6 * 1e+3)

    disc_df = pd.read_csv(discriminators_losses_file_path)
    gen_df = pd.read_csv(generators_losses_file_path)

    wanted_disc_df = disc_df.iloc[plot_after:plot_before]
    wanted_gen_df = gen_df.iloc[plot_after:plot_before]

    wanted_disc_df.plot(x=0, y=1, kind='scatter', title="Discriminator", ylabel='Loss', xlabel="Iteration")
    wanted_gen_df.plot(x=0, y=1, kind='scatter', title='Generator', ylabel='Loss', xlabel="Iteration")

    plt.show()

