import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

male = 'male.flac'
male_to_female = 'male-to-female.flac'
female = 'female.flac'
dir_path = 'C:\\Users\\Konrad Kulesza\\Desktop\\prezentacja'

current = male_to_female

file_path = os.path.join(dir_path, current)

if __name__ == '__main__':
    signal, sr = librosa.load(file_path, duration=4)
    mel_spectogram = librosa.feature.melspectrogram(signal, sr=sr, n_fft=2048, hop_length=512, n_mels=90)
    mel_spectogram = librosa.power_to_db(mel_spectogram)
    librosa.display.specshow(mel_spectogram,
                             x_axis="time",
                             y_axis="mel",
                             sr=sr
                             )
    plt.colorbar(format="%+2.f")
    plt.title(current)
    plt.show()
