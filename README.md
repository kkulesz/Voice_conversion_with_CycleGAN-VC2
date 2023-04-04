# Voice Conversion with `Cycle-GAN-VC`s

This repository contains code, data and other artifacts used or produced during working on my bachelor's thesis, especially:
- `PyTorch` and `Pytorch Lightning` code for Cycle-GAN-VC [1] and Cycle-GAN-VC2 [2] models (`src/model/`).
- code for data preprocessing (`src/data_preprocessing/`). The code was based on [3], but rewritten in more OOP way, so it was easier to use for me. 
- python scripts for training and other minor tasks (`scripts/`).
- selected speakers data from `LibriSpeech` [4] and `VC-challange-2016` [5] datasets (`data/`).


## Usage
0. install `requirements.txt`
1. download your data - you can use `data` directory in this repository, run `download_vc16_dataset.py` in order to download `VC-challange-2016` dataset or just get your own data.
2. modify paths and hiperparameters in `consts.py` in order to achieve your goal.
3. run `prepare_directories_and_preprocess.py`.
4. run `train_vanilla.py` or `train_lightning.py`. Second one uses model written in `PyTorch Lightning`.
5. after its done, you can validate your training process and model using `plot_losses.py`, `plot_mel_spectogram.py` and `validate_trained_model` scripts.


## Reference
1. T. Kaneko, H. Kameoka, Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks, 2017. eprint: 1711.11293
2. T. Kaneko, H. Kameoka, K. Tanaka i N. Hojo, CycleGAN-VC2: Improved CycleGAN based Non-parallel Voice Conversion, 2019. arXiv: 1904.04631 [cs.SD].
3. P. Yuvraj, https://github.com/pritishyuvraj/Voice-Conversion-GAN.
4. V. Panayotov, G. Chen, D. Povey i S. Khudanpur, „Librispeech: An ASR corpus based on public domain audio books”, w 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2015, s. 5206–5210. DOI: 10.1109/ICASSP.
2015.7178964.
5. T. Toda, L.-H. Chen, D. Saito i in., „The Voice Conversion Challenge 2016”, w Proc. Interspeech 2016, 2016, s. 1632–1636. DOI: 10.21437/Interspeech.2016-1066.
