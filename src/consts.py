import os


class Consts:
    # preprocessing constants
    f0_floor = 71.0
    f0_ceil = 800.0
    number_of_mcpes = 24
    sampling_rate = 16000
    frame_period_in_ms = 5.0
    number_of_frames = 128

    # directories
    project_dir_path = "E:\\STUDIA\\inzynierka\\2_moje_przygotowania\\3.kod\\moje_repo\\src\\"

    # train directories
    data_dir = os.path.join(project_dir_path, "data")
    vc16_data_dir = os.path.join(data_dir, "vc-challenge-2016\\vcc2016_training")
    male_1 = "SM1"
    male_2 = "SM2"
    female_1 = "SF1"
    female_2 = "SF2"
    male_to_female = (male_1, female_1)
    male_to_male = (male_1, male_2)
    female_to_male = (female_1, male_1)
    female_to_female = (female_1, female_2)

    # cache directories
    cache_dir = os.path.join(project_dir_path, "cache")
    A_cache_dir = os.path.join(cache_dir, "A")
    B_cache_dir = os.path.join(cache_dir, "B")
    mcep_file = "mcep_normalization.npz"
    f0_file = "f0_normalization.npz"
    spectral_envelope_file = "spectral_envelope.pickle"
    A_preprocessed_dataset_file = os.path.join(A_cache_dir, spectral_envelope_file)
    B_preprocessed_dataset_file = os.path.join(B_cache_dir, spectral_envelope_file)

