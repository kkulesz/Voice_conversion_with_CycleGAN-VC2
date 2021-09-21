from src.consts import Consts

from src.preprocessing.prepare_project_directory import prepare_project_dir
from src.preprocessing.pyworld_preprocessor import PyWorldPreprocessor

from src.dataset.preprocessed_dataset import PreprocessedDataset

from src.model.generator import Generator
from src.model.discriminator import Discriminator

if __name__ == '__main__':
    # testing pipeline

    prepare_project_dir()
    preprocessor = PyWorldPreprocessor(number_of_mceps=Consts.number_of_mcpes,
                                       sampling_rate=Consts.sampling_rate,
                                       frame_period_in_ms=Consts.frame_period_in_ms)

    A_dir, B_dir = Consts.male_to_female
    preprocessor.preprocess(data_directory=Consts.vc16_data_dir,
                            A_dir=A_dir,
                            B_dir=B_dir,
                            cache_directory=Consts.cache_dir)

    dataset = PreprocessedDataset(A_dataset_file=Consts.A_preprocessed_dataset_file,
                                  B_dataset_file=Consts.B_preprocessed_dataset_file,
                                  number_of_frames=Consts.number_of_frames)