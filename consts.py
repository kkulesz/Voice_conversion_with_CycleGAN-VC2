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
    data_dir = os.path.join(project_dir_path, "src\\data")
    data_dir_vc16 = os.path.join(data_dir, "vc-challenge-2016\\vcc2016_training")
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

    cache_dir = os.path.join(output_dir_path, "cache")
    A_cache_dir = os.path.join(cache_dir, A_dir_name)
    B_cache_dir = os.path.join(cache_dir, B_dir_name)
    mcep_norm_file = "mcep_normalization.npz"
    log_f0_norm_file = "log_f0_normalization.npz"
    spectral_envelope_file = "mcep.pickle"
    A_preprocessed_dataset_file = os.path.join(A_cache_dir, spectral_envelope_file)
    B_preprocessed_dataset_file = os.path.join(B_cache_dir, spectral_envelope_file)

    # ------------------------------ #
    #  validation                    #
    # ------------------------------ #
    validation_output_dir = os.path.join(output_dir_path, "validation")
    A_val_dir_name = "AtoB"
    B_val_dir_name = "BtoA"
    A_output_dir = os.path.join(validation_output_dir, A_val_dir_name)
    B_output_dir = os.path.join(validation_output_dir, B_val_dir_name)
    dump_validation_file_epoch_frequency = 1
    print_losses_iteration_frequency = 50

