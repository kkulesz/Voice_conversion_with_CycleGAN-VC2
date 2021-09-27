import os


class Consts:
    # ------------------------------ #
    #  data processing params        #
    # ------------------------------ #
    f0_floor = 71.0
    f0_ceil = 800.0
    number_of_mcpes = 24
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
    project_dir_path = "E:\\STUDIA\\inzynierka\\2_moje_przygotowania\\3.kod\\moje_repo\\"
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

    male_source = "SM1"
    male_target = "TM1"
    female_source = "SF1"
    female_target = "TF1"
    male_to_female = (male_source, female_target)
    male_to_male = (male_source, male_target)
    female_to_male = (female_source, male_target)
    female_to_female = (female_source, female_target)

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
    A_validation_output_directory_name = "AtoB"
    B_validation_output_directory_name = "BtoA"
    A_validation_output_directory_path = os.path.join(validation_output_directory, A_validation_output_directory_name)
    B_validation_output_directory_path = os.path.join(validation_output_directory, B_validation_output_directory_name)
    dump_validation_file_epoch_frequency = 50
    print_losses_iteration_frequency = 10
