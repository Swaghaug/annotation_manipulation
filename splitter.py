import os
import math
import argparse
from PIL import Image
from shapely.geometry import Polygon
from shapely.affinity import translate

def split_image_and_annotations(
    image_path,
    label_path,
    output_dir,
    tile_width=640,
    tile_height=640
):
    """
    Splits a given large image into smaller tiles (tile_width x tile_height),
    and clips polygon annotations accordingly.

    :param image_path: Path to the original image.
    :param label_path: Path to the corresponding polygon annotation file.
    :param output_dir: Directory where split images and labels will be saved.
    :param tile_width: Width of each tile.
    :param tile_height: Height of each tile.
    """

    # 1) Create output directories if they don't exist
    images_out_dir = os.path.join(output_dir, "images")
    labels_out_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    # 2) Load image
    image = Image.open(image_path)
    img_width, img_height = image.size
    print(f"Processing: {os.path.basename(image_path)} | Size: {img_width}x{img_height}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 3) Read and parse annotations (polygon YOLO format)
    annotations = []
    try:
        with open(label_path, "r") as f:
            for line in f:
                items = line.strip().split()
                if len(items) < 6:  # Minimum of 3 points + class_id
                    print(f"Skipping incomplete annotation: {line.strip()}")
                    continue
                class_id = items[0]
                coords = list(map(float, items[1:]))

                # Convert from normalized [0..1] to absolute pixel coords
                points = [
                    (coords[i] * img_width, coords[i + 1] * img_height)
                    for i in range(0, len(coords), 2)
                ]
                # Build polygon
                polygon = Polygon(points)

                # Attempt to fix minor geometry issues
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    print(f"Skipping invalid polygon (still invalid after fix): {line.strip()}")
                    continue

                annotations.append((class_id, polygon))
    except FileNotFoundError:
        print(f"No label found at {label_path}. We'll still save tiles, but no annotation data will exist.")
        # We do NOT return here, because we still want to generate tile images (empty annotation)

    # 4) Generate tiles
    row_count = math.ceil(img_height / tile_height)
    col_count = math.ceil(img_width / tile_width)

    for row_idx in range(row_count):
        for col_idx in range(col_count):
            tile_left = col_idx * tile_width
            tile_top = row_idx * tile_height
            tile_right = min(tile_left + tile_width, img_width)
            tile_bottom = min(tile_top + tile_height, img_height)

            # Crop image
            tile_image = image.crop((tile_left, tile_top, tile_right, tile_bottom))

            # Prepare tile bounding polygon (for clipping)
            tile_polygon = Polygon([
                (tile_left, tile_top),
                (tile_right, tile_top),
                (tile_right, tile_bottom),
                (tile_left, tile_bottom)
            ])

            # 5) For each annotation polygon, clip with tile_polygon
            clipped_annotations = []
            for class_id, polygon in annotations:
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    continue
                clipped_poly = polygon.intersection(tile_polygon)
                if not clipped_poly.is_empty:
                    if clipped_poly.geom_type == 'MultiPolygon':
                        for subpoly in clipped_poly.geoms:
                            shifted = _shift_polygon(subpoly, tile_left, tile_top)
                            if not shifted.is_empty:
                                clipped_annotations.append((class_id, shifted))
                    elif clipped_poly.geom_type == 'Polygon':
                        shifted = _shift_polygon(clipped_poly, tile_left, tile_top)
                        if not shifted.is_empty:
                            clipped_annotations.append((class_id, shifted))

            # 6) Always save tile image & label (even if empty)
            #    row_idx+1, col_idx+1 => 1-based indexing
            #    :02d => zero-pad to 2 digits
            tile_image_name = f"{base_name}_{(row_idx):02d}_{(col_idx):02d}.jpg"
            tile_label_name = f"{base_name}_{(row_idx):02d}_{(col_idx):02d}.txt"

            tile_image.save(os.path.join(images_out_dir, tile_image_name))

            tile_label_path = os.path.join(labels_out_dir, tile_label_name)
            with open(tile_label_path, 'w') as lf:
                if clipped_annotations:
                    for cls_id, poly in clipped_annotations:
                        coords = list(poly.exterior.coords)
                        coords_str = ' '.join([
                            f"{(p[0] / (tile_right - tile_left)):.6f} {(p[1] / (tile_bottom - tile_top)):.6f}"
                            for p in coords[:-1]
                        ])
                        lf.write(f"{cls_id} {coords_str}\n")
                # else leave empty

def _shift_polygon(polygon, shift_x, shift_y):
    """Shift the polygon's coordinates by subtracting shift_x and shift_y."""
    return translate(polygon, xoff=-shift_x, yoff=-shift_y)

def split_images_in_folder(
    input_dir,
    output_dir,
    tile_width=640,
    tile_height=640,
    valid_exts=(".png", ".jpg", ".jpeg")
):
    """
    Splits all images in 'input_dir/images' into tiles, with corresponding labels
    from 'input_dir/labels'. The results go into 'output_dir/images' & 'output_dir/labels'.
    """
    images_dir = os.path.join(input_dir, "images")
    labels_dir = os.path.join(input_dir, "labels")

    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    if not os.path.exists(labels_dir):
        print(f"Labels directory not found: {labels_dir}")
        return
    
    # List valid images
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
    if not all_images:
        print(f"No images found in {images_dir} with extensions {valid_exts}")
        return
    
    all_images.sort()
    os.makedirs(output_dir, exist_ok=True)

    for img_file in all_images:
        image_path = os.path.join(images_dir, img_file)
        base_name, _ = os.path.splitext(img_file)
        label_path = os.path.join(labels_dir, base_name + ".txt")

        split_image_and_annotations(
            image_path=image_path,
            label_path=label_path,
            output_dir=output_dir,
            tile_width=tile_width,
            tile_height=tile_height
        )

def main():
    parser = argparse.ArgumentParser(description="Split images into tiles and clip annotations.")
    parser.add_argument(
        "--input_dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing subfolders 'images/' and 'labels/' to be processed."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_dataset/output",
        help="Directory where split tiles will be saved. Default: 'test_dataset/output'"
    )
    parser.add_argument(
        "--tile_width",
        type=int,
        default=640,
        help="Tile width in pixels. Default: 640"
    )
    parser.add_argument(
        "--tile_height",
        type=int,
        default=640,
        help="Tile height in pixels. Default: 640"
    )

    args = parser.parse_args()

    # Process each input_dir in turn
    for directory in args.input_dir:
        print(f"\n===== Splitting images in directory: {directory} =====")
        split_images_in_folder(
            input_dir=directory,
            output_dir=args.output_dir,
            tile_width=args.tile_width,
            tile_height=args.tile_height
        )

if __name__ == "__main__":
    main()
