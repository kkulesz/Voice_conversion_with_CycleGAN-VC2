import os
import zipfile
from urllib.request import urlretrieve

from consts import Consts


def _check_if_already_downloaded(path):
    return os.path.exists(path)


def _unzip(zipped_file_path, destination_dir):
    unzipped_directory_name = zipped_file_path[:-len(".zip")]
    if os.path.exists(unzipped_directory_name):
        print(f"- already unzipped: {unzipped_directory_name}")
    else:
        print(f"- unzipping: {zipped_file_path}")
        with zipfile.ZipFile(zipped_file_path) as zipped_file:
            zipped_file.extractall(destination_dir)
        print(f"- unzipped: {unzipped_directory_name}")


def _download(url, file_path):
    if not _check_if_already_downloaded(file_path):
        print(f"Downloading: {url}")
        urlretrieve(url, file_path)
        print(f"- finished downloading: {file_path}")
    else:
        print(f"- already downloaded: {file_path}")


def download_vc2016_dataset(destination_directory):
    for file_to_download in Consts.vc16_download_directories:
        download_url = Consts.vc16_url_prefix + file_to_download
        file_path = os.path.join(destination_directory, file_to_download)

        _download(download_url, file_path)
        _unzip(file_path, destination_directory)
