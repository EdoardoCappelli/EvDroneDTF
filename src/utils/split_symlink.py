import os
import shutil

def process_split(txt_file, target_dir):
    """
    Reads a text file containing folder names (e.g., "8/ 9/")
    and symlinks them into target_dir.
    """
    # 1. Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    # 2. Read the split file
    try:
        with open(txt_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {txt_file}")
        return
    
    path_prefix = txt_file.split('/')[:-1]
    path_prefix = '/'.join(path_prefix) + '/'
    print(f"Using path prefix: {path_prefix}")

    # 3. Parse entries (handles spaces, newlines, and tabs automatically)
    folder_entries = content.split()

    print(f"Processing {len(folder_entries)} items from {txt_file}...")

    count = 0
    for entry in folder_entries:
        # Remove trailing slashes (turns "8/" into "8")
        folder_name = entry.strip('/')
        
        # Define Source and Destination
        # We use abspath to ensure links work even if you move the train/test folders later
        src_path = os.path.abspath(path_prefix + folder_name)
        dst_path = os.path.join(target_dir, folder_name)

        # Check if the source folder actually exists
        if not os.path.exists(src_path):
            print(f"Warning: Source folder '{folder_name}' not found. Skipping.")
            continue

        # Create Symlink
        try:
            os.symlink(src_path, dst_path)
            count += 1
        except FileExistsError:
            print(f"Note: Link for '{folder_name}' already exists in {target_dir}.")
        except OSError as e:
            print(f"Error linking {folder_name}: {e}")

    print(f"Success: Symlinked {count} folders into '{target_dir}'.")
    print("-" * 30)

def main():
    # Define your input files and output directories
    splits = [
        ("/home/gmagrini/datasets/FRED/train_split.txt", "/home/gmagrini/datasets/FRED_split/train"),
        ("/home/gmagrini/datasets/FRED/test_split.txt", "/home/gmagrini/datasets/FRED_split/test")
    ]

    for txt_file, output_folder in splits:
        print(f"Starting process for {txt_file}...")
        process_split(txt_file, output_folder)

if __name__ == "__main__":
    main()