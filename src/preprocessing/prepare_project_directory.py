import os
import shutil

from src.consts import Consts


def create_cache_files(directory):
    A_f0_file = os.path.join(directory, Consts.f0_file)
    A_mcep_file = os.path.join(directory, Consts.mcep_file)
    A_spectral_envelope_file = os.path.join(directory, Consts.spectral_envelope_file)
    open(A_f0_file, "w")
    open(A_mcep_file, "w")
    open(A_spectral_envelope_file, "w")


def create_cache_dirs():
    os.mkdir(Consts.cache_dir)

    os.mkdir(Consts.A_cache_dir)
    os.mkdir(Consts.B_cache_dir)

    create_cache_files(Consts.A_cache_dir)
    create_cache_files(Consts.B_cache_dir)


def delete_cache_dirs():
    shutil.rmtree(Consts.cache_dir)


def prepare_project_dir():
    delete_cache_dirs()
    create_cache_dirs()


if __name__ == '__main__':
    prepare_project_dir()
