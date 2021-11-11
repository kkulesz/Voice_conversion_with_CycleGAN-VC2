import os
import pandas as pd
import matplotlib.pyplot as plt

from consts import Consts

if __name__ == '__main__':
    directory_path = Consts.models_storage_directory_path
    discriminators_losses_file_path = os.path.join(directory_path, Consts.discriminator_loss_storage_file)
    generators_losses_file_path = os.path.join(directory_path, Consts.generator_loss_storage_file)

    plot_after = int(0 * 1e+3)

    disc_df = pd.read_csv(discriminators_losses_file_path)
    gen_df = pd.read_csv(generators_losses_file_path)

    wanted_disc_df = disc_df.iloc[plot_after:]
    wanted_gen_df = gen_df.iloc[plot_after:]

    wanted_disc_df.plot(x=0, y=1, kind='scatter', title="Discriminator", ylabel='Loss', xlabel="Iteration")
    wanted_gen_df.plot(x=0, y=1, kind='scatter', title='Generator', ylabel='Loss', xlabel="Iteration")

    plt.show()
