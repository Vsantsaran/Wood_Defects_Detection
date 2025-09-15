import os
import shutil
import random

def split_data(images_dir, annotations_dir, output_root_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """
    Splits image and annotation files into train, validation, and test sets.

    Args:
        images_dir (str): Path to the directory containing image files.
        annotations_dir (str): Path to the directory containing annotation files (.txt).
        output_root_dir (str): The root directory where 'train', 'val', 'test' folders will be created.
        train_ratio (float): Proportion of data for the training set.
        val_ratio (float): Proportion of data for the validation set.
        test_ratio (float): Proportion of data for the test set.
    """

    # Ensure ratios sum to 1
    if not (round(train_ratio + val_ratio + test_ratio) == 1.0):
        print("Error: train_ratio, val_ratio, and test_ratio must sum to 1.0")
        return

    # Get list of all image files (assuming image files and annotation files have same base names)
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    random.shuffle(image_files) # Shuffle the list to ensure random splitting

    total_images = len(image_files)
    print(f"Total images found: {total_images}")

    # Calculate split sizes
    train_count = int(total_images * train_ratio)
    val_count = int(total_images * val_ratio)
    test_count = total_images - train_count - val_count # Ensure all images are included

    print(f"Splitting into: Train={train_count}, Val={val_count}, Test={test_count}")

    # Split the image files
    train_files = image_files[:train_count]
    val_files = image_files[train_count : train_count + val_count]
    test_files = image_files[train_count + val_count :]

    # Define the output structure
    sets = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    # Create the output directories
    for set_name in sets.keys():
        os.makedirs(os.path.join(output_root_dir, set_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_root_dir, set_name, "labels"), exist_ok=True)
        print(f"Created directory: {os.path.join(output_root_dir, set_name, 'images')}")
        print(f"Created directory: {os.path.join(output_root_dir, set_name, 'labels')}")

    # Copy files to their respective destinations
    for set_name, files_to_copy in sets.items():
        print(f"\nCopying {set_name} files...")
        for img_file in files_to_copy:
            base_name = os.path.splitext(img_file)[0]
            annotation_file = base_name + ".txt"

            # Source paths
            src_image_path = os.path.join(images_dir, img_file)
            src_annotation_path = os.path.join(annotations_dir, annotation_file)

            # Destination paths
            dest_image_path = os.path.join(output_root_dir, set_name, "images", img_file)
            dest_annotation_path = os.path.join(output_root_dir, set_name, "labels", annotation_file)

            try:
                shutil.copy2(src_image_path, dest_image_path)
                shutil.copy2(src_annotation_path, dest_annotation_path)
            except FileNotFoundError as e:
                print(f"Warning: Could not copy {e}. Ensure all image files have corresponding annotation files.")
            except Exception as e:
                print(f"An error occurred while copying {img_file}: {e}")

    print("\nData splitting complete!")
    print(f"Train images: {len(train_files)}, labels: {len(train_files)}")
    print(f"Validation images: {len(val_files)}, labels: {len(val_files)}")
    print(f"Test images: {len(test_files)}, labels: {len(test_files)}")

# --- Usage Example ---
if __name__ == "__main__":
    # Define your input directories
    # IMPORTANT: Replace these with the actual paths to your image and annotation folders
    input_images_directory = 'datasets/augmented_ds/images'      # e.g., 'Dataset/images'
    input_annotations_directory = 'datasets/augmented_ds/labels' # e.g., 'yolov8_annotations'

    # Define the root directory where the output structure will be created
    output_dataset_root = 'splitted_data'

    split_data(input_images_directory, input_annotations_directory, output_dataset_root)
