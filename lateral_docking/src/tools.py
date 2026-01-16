import os

def count_files_in_directory(directory_path) -> tuple[int]:
    """Counts the number of files in the specified directory."""
    try:
        raw_data_count = len([name for name in os.listdir(directory_path+f"raw_data/") if os.path.isfile(os.path.join(directory_path+f"raw_data/", name))])
        detector_data_count = len([name for name in os.listdir(directory_path+f"output_data/") if os.path.isfile(os.path.join(directory_path+f"output_data/", name))])
        return (raw_data_count, detector_data_count, )
    except FileNotFoundError:
        print(f"The directory {directory_path} does not exist.")
        exit()