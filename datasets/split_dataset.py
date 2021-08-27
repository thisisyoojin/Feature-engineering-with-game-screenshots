import os
import shutil

    
def make_dirs_per_dataset(root_dir, sub_dirs):
    """
    Create directories for dividing images with train/validation/test datasets
    """
    for dir in sub_dirs:
        os.makedirs(f"{root_dir}/train/{dir}", exist_ok=True)
        os.makedirs(f"{root_dir}/validation/{dir}", exist_ok=True)
        os.makedirs(f"{root_dir}/test/{dir}", exist_ok=True)


def copy_files(root_dir, sub_dirs):
    """
    Split the data for train/validation/test dataset
    (train data: 0.6, validation data: 0.2, test data: 0.2)
    """
    def folder_name(type, folder):
        return os.path.join(root_dir, type, folder)

    for folder in sub_dirs:
        all_path = folder_name('all', folder)
        files = os.listdir(all_path)
        for file in files:
            idx = int(file.split('_')[1].split('.')[0])
            if idx < 3:
                shutil.copyfile(f"{all_path}/{file}", f"{folder_name('train', folder)}/{file}")
            elif idx == 3:
                shutil.copyfile(f"{all_path}/{file}", f"{folder_name('validation', folder)}/{file}")
            else:
                shutil.copyfile(f"{all_path}/{file}", f"{folder_name('test', folder)}/{file}")


def data_split(root_dir):
    """
    Split image files in a designated folder
    """
    sub_dirs = os.listdir(f"{root_dir}/all")
    make_dirs_per_dataset(root_dir, sub_dirs)
    copy_files(root_dir, sub_dirs)