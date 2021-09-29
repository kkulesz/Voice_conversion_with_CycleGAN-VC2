import os
import pandas as pd
import matplotlib.pyplot as plt

from consts import Consts

if __name__ == '__main__':
    directory_path = Consts.models_storage_directory_path
    discriminators_losses_file_path = os.path.join(directory_path, Consts.discriminator_loss_storage_file)
    generators_losses_file_path = os.path.join(directory_path, Consts.generator_loss_storage_file)

    disc_df = pd.read_csv(discriminators_losses_file_path)
    gen_df = pd.read_csv(generators_losses_file_path)

    disc_df.plot(x=0, y=1, kind='scatter', title="Discriminator")
    gen_df.plot(x=0, y=1, kind='scatter', title='Generator')

    plt.show()

