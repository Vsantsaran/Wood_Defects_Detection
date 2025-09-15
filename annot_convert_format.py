import json
import os

def coco_to_yolov8(coco_json_path, output_dir):
    """
    Converts COCO format annotations to YOLOv8 format,
    creating a file for every image, even those with no annotations.

    Args:
        coco_json_path (str): Path to the COCO formatted JSON file.
        output_dir (str): Directory where the YOLOv8 annotation files will be saved.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the COCO JSON file
    try:
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: COCO JSON file not found at {coco_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {coco_json_path}. Check file integrity.")
        return

    # Create a dictionary to map image IDs to their metadata (file_name, width, height)
    # This ensures we have info for ALL images, not just those with annotations.
    image_id_to_info = {
        img['id']: {'file_name': img['file_name'], 'width': img['width'], 'height': img['height']}
        for img in coco_data['images']
    }

    # Group annotations by image ID
    annotations_by_image_id = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image_id:
            annotations_by_image_id[image_id] = []
        annotations_by_image_id[image_id].append(ann)

    print(f"Processing {len(coco_data['images'])} images...")

    # Iterate through ALL images to ensure every image gets an annotation file
    for image_id, img_info in image_id_to_info.items():
        img_width = img_info['width']
        img_height = img_info['height']
        img_file_name = img_info['file_name']

        # Construct the output file path (e.g., 100000000.txt)
        output_file_name = os.path.splitext(img_file_name)[0] + '.txt'
        output_file_path = os.path.join(output_dir, output_file_name)

        yolov8_lines = []
        # Check if there are annotations for the current image
        if image_id in annotations_by_image_id:
            for ann in annotations_by_image_id[image_id]:
                category_id = ann['category_id']
                # COCO bbox format: [x_min, y_min, width, height]
                x_min, y_min, bbox_width, bbox_height = ann['bbox']

                # Calculate YOLOv8 format: [class_id, x_center, y_center, width, height]
                # Normalize coordinates by image width and height
                x_center = (x_min + bbox_width / 2) / img_width
                y_center = (y_min + bbox_height / 2) / img_height
                normalized_width = bbox_width / img_width
                normalized_height = bbox_height / img_height

                # Append the formatted line
                yolov8_lines.append(
                    f"{category_id} {x_center:.6f} {y_center:.6f} {normalized_width:.6f} {normalized_height:.6f}"
                )

        # Write the annotations (or an empty string if no annotations) to the output file
        with open(output_file_path, 'w') as f:
            f.write("\n".join(yolov8_lines))
        # print(f"Generated annotation file for {img_file_name}")

    print(f"\nConversion complete! YOLOv8 annotation files saved to: {output_dir}")
    print(f"Total images processed: {len(image_id_to_info)}")
    print(f"Total annotation files created: {len(os.listdir(output_dir))}")

# --- Usage Example ---
if __name__ == "__main__":
    # Define the path to your COCO JSON file
    # Ensure 'bbox_coco_dataset.json' is in the same directory as this script,
    # or provide the full path to it.
    coco_json_file = 'wood_defects/new_work/Dataset2/bbox_coco_dataset.json'

    # Define the output directory for YOLOv8 annotation files
    output_annotations_dir = 'yolov8_annotations'

    coco_to_yolov8(coco_json_file, output_annotations_dir)
