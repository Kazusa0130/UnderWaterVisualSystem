import os
from datetime import datetime

def count_files_in_directory(directory_path):
    """Counts the number of files in the specified directory."""
    raw_data_dir = os.path.join(directory_path, "raw_data")
    output_data_dir = os.path.join(directory_path, "output_data")
    traj_data_dir = os.path.join(directory_path, "traj_data")

    for d in [raw_data_dir, output_data_dir, traj_data_dir]:
        os.makedirs(d, exist_ok=True)

    try:
        raw_data_count = len([name for name in os.listdir(raw_data_dir) if os.path.isfile(os.path.join(raw_data_dir, name))])
        detector_data_count = len([name for name in os.listdir(output_data_dir) if os.path.isfile(os.path.join(output_data_dir, name))])
        traj_data_count = len([name for name in os.listdir(traj_data_dir) if os.path.isfile(os.path.join(traj_data_dir, name))])
        return (raw_data_count, detector_data_count, traj_data_count)
    except Exception as e:
        print(f"Error counting files in {directory_path}: {e}")
        return (0, 0, 0)


def get_latest_traj_file(directory_path):
    """Get the most recently modified trajectory file.

    Args:
        directory_path: Base output directory path (should contain traj_data/ subdir).

    Returns:
        Path to the most recently modified trajectory file, or None if no files exist.
    """
    traj_dir = os.path.join(directory_path, "traj_data")
    if not os.path.exists(traj_dir):
        return None

    files = [os.path.join(traj_dir, f) for f in os.listdir(traj_dir)
             if os.path.isfile(os.path.join(traj_dir, f)) and f.endswith('.csv')]
    if not files:
        return None

    return max(files, key=os.path.getmtime)


def parse_traj_file(filepath):
    """Parse a trajectory CSV file and extract pose data.

    File format:
        timestamp,frame_id,x,y,z,yaw,pitch,roll

    Args:
        filepath: Path to the trajectory CSV file.

    Returns:
        Tuple of (timestamps, tvecs, rvecs, valid_mask) where each is a numpy array.
        valid_mask is inferred from whether x,y,z,yaw,pitch,roll are all zero.
        Returns None if file cannot be parsed.
    """
    import numpy as np

    timestamps = []
    tvecs = []
    rvecs = []
    valid_mask = []

    try:
        with open(filepath, 'r') as f:
            # Skip header if present
            first_line = f.readline().strip()
            if not first_line.startswith('timestamp'):
                # It's data, parse it
                parts = first_line.split(',')
                if len(parts) >= 8:
                    timestamps.append(float(parts[0]))
                    tvecs.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    # Restore rvec from roll,pitch,yaw ordering: rvec = [roll, pitch, yaw]
                    rvecs.append([float(parts[7]), float(parts[6]), float(parts[5])])
                    is_valid = not (float(parts[2]) == 0.0 and float(parts[3]) == 0.0 and float(parts[4]) == 0.0
                                    and float(parts[5]) == 0.0 and float(parts[6]) == 0.0 and float(parts[7]) == 0.0)
                    valid_mask.append(is_valid)

            # Read remaining lines
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 8:
                    timestamps.append(float(parts[0]))
                    tvecs.append([float(parts[2]), float(parts[3]), float(parts[4])])
                    # Restore rvec from roll,pitch,yaw ordering: rvec = [roll, pitch, yaw]
                    rvecs.append([float(parts[7]), float(parts[6]), float(parts[5])])
                    is_valid = not (float(parts[2]) == 0.0 and float(parts[3]) == 0.0 and float(parts[4]) == 0.0
                                    and float(parts[5]) == 0.0 and float(parts[6]) == 0.0 and float(parts[7]) == 0.0)
                    valid_mask.append(is_valid)

        return (np.array(timestamps),
                np.array(tvecs),
                np.array(rvecs),
                np.array(valid_mask))
    except Exception as e:
        print(f"Error parsing trajectory file: {e}")
        return None