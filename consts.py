import os


class Consts:
    # ------------------------------ #
    #  data processing params        #
    # ------------------------------ #
    f0_floor = 71.0
    f0_ceil = 800.0
    cycle_gan_1_mceps = 24
    cycle_gan_2_mceps = 36
    number_of_mceps = cycle_gan_2_mceps  # change here to preprocess and validate files correctly
    sampling_rate = 16000
    frame_period_in_ms = 5.0
    number_of_frames = 128

    # ------------------------------ #
    #  training params               #
    # ------------------------------ #
    mini_batch_size = 1
    generator_lr = 0.0002
    discriminator_lr = 0.0001
    generator_lr_decay = generator_lr / 200000
    discriminator_lr_decay = discriminator_lr / 200000
    start_decay_after = 200000
    adam_optimizer_betas = (0.5, 0.999)
    cycle_loss_lambda = 10
    identity_loss_lambda = 5
    zero_identity_loss_lambda_after = 10000
    number_of_epochs = 5000

    # ------------------------------ #
    #  FILES AND DIRECTORIES         #
    # ------------------------------ #
    project_dir_path = os.path.dirname(os.path.abspath(__file__))
    output_dir_path = os.path.join(project_dir_path, "output")
    A_dir_name = "A"
    B_dir_name = "B"

    # ------------------------------ #
    #  train directories             #
    # ------------------------------ #
    data_directory_path = os.path.join(project_dir_path, "data")

    vc16_data_directory_path = os.path.join(data_directory_path, "vc-challenge-2016")
    vc16_training_directory_path = os.path.join(vc16_data_directory_path, "vcc2016_training")
    vc16_validation_directory_path = os.path.join(vc16_data_directory_path, "evaluation_all")
    vc16_url_prefix = "https://datashare.is.ed.ac.uk/bitstream/handle/10283/2211/"
    vc16_download_directories = ["vcc2016_training.zip", "evaluation_all.zip"]

    male_1 = "SM1"
    male_2 = "SM2"
    female_1 = "SF1"
    female_2 = "SF2"
    male_to_female = (male_1, female_1)
    male_to_male = (male_1, male_2)
    female_to_male = (female_1, male_1)
    female_to_female = (female_1, female_2)

    # ------------------------------ #
    #  cache directories             #
    # ------------------------------ #
    cache_directory_path = os.path.join(output_dir_path, "cache")
    A_cache_directory_path = os.path.join(cache_directory_path, A_dir_name)
    B_cache_directory_path = os.path.join(cache_directory_path, B_dir_name)
    mcep_norm_filename = "mcep_normalization.npz"
    log_f0_norm_filename = "log_f0_normalization.npz"
    spectral_envelope_filename = "mcep.pickle"
    A_preprocessed_dataset_file_path = os.path.join(A_cache_directory_path, spectral_envelope_filename)
    B_preprocessed_dataset_file_path = os.path.join(B_cache_directory_path, spectral_envelope_filename)

    # ------------------------------ #
    #  validation                    #
    # ------------------------------ #
    validation_output_directory = os.path.join(output_dir_path, "validation")
    log_file_path = os.path.join(output_dir_path, "log_file.txt")
    A2B_validation_output_directory_name = "AtoB"
    B2A_validation_output_directory_name = "BtoA"
    A2B_validation_output_directory_path = os.path.join(validation_output_directory, A2B_validation_output_directory_name)
    B2A_validation_output_directory_path = os.path.join(validation_output_directory, B2A_validation_output_directory_name)
    dump_validation_file_epoch_frequency = 500
    print_losses_iteration_frequency = 100

    # ------------------------------ #
    #  model storage                 #
    # ------------------------------ #
    models_saving_epoch_frequency = 100
    models_storage_directory_path = os.path.join(output_dir_path, "models_storage")

    A2B_generator_file_name = "A2B_gen.pth"
    B2A_generator_file_name = "B2A_gen.pth"
    A_discriminator_file_name = "A_disc.pth"
    B_discriminator_file_name = "B_disc.pth"

    generator_loss_storage_file = "generators_loss.csv"
    discriminator_loss_storage_file = "discriminators_loss.csv"

    # ------------------------------ #
    #  my storage                    #
    # ------------------------------ #
    my_storage = os.path.join(project_dir_path, "storage")
    submodules_graph_storage = os.path.join(my_storage, "submodules_graph")

