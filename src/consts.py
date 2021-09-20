import os


class Consts:
    # preprocessing constants
    f0_floor = 71.0
    f0_ceil = 800.0

    # directories
    project_dir_path = "E:\\STUDIA\\inzynierka\\2_moje_przygotowania\\3.kod\\moje_repo\\src\\"

    data_dir = os.path.join(project_dir_path, "data")
    vc16_data_dir = os.path.join(data_dir, "vc-challenge-2016\\vcc2016_training")
    male_1 = "SM1"
    male_2 = "SM2"
    female_1 = "SF1"
    female_2 = "SF2"

    cache_dir = os.path.join(project_dir_path, "cache")
    A_cache_dir = os.path.join(cache_dir, "A")
    B_cache_dir = os.path.join(cache_dir, "B")
    mcep_file = "mcep_normalization.npz"
    f0_file = "f0_normalization.npz"
    spectral_envelope_file = "spectral_envelope.pickle"

