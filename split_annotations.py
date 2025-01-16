import os
from PIL import Image
from shapely.geometry import Polygon
from shapely.affinity import translate
import math

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
    print(f"Image size: {img_width}x{img_height}")
    print("Tile size:", tile_width, "x", tile_height)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 3) Read and parse annotations (polygon YOLO format)
    annotations = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            items = line.strip().split()
            if len(items) < 6:  # Minimum of 3 points (x1, y1, x2, y2, x3, y3)
                print(f"Not enough vertex points to split annotation in {line.strip()}")
                continue
            
            class_id = items[0]
            coords = list(map(float, items[1:]))
            points = [(coords[i] * img_width, coords[i + 1] * img_height) for i in range(0, len(coords), 2)]
            annotations.append((class_id, Polygon(points)))

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

            # Prepare tile polygons
            tile_polygon = Polygon([
                (tile_left, tile_top),
                (tile_right, tile_top),
                (tile_right, tile_bottom),
                (tile_left, tile_bottom)
            ])

            # 5) For each annotation polygon, clip with tile polygon
            clipped_annotations = []
            for class_id, polygon in annotations:
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

            # 6) Save tiles and annotations
            if clipped_annotations:
                tile_image_name = f"{base_name}_{row_idx}_{col_idx}.jpg"
                tile_label_name = f"{base_name}_{row_idx}_{col_idx}.txt"

                # Save tile image
                tile_image.save(os.path.join(images_out_dir, tile_image_name))

                # Save tile label
                with open(os.path.join(labels_out_dir, tile_label_name), 'w') as lf:
                    for class_id, poly in clipped_annotations:
                        coords = list(poly.exterior.coords)
                        # Re-normalize to the tile size
                        coords_str = ' '.join([f"{(p[0] / tile_width):.6f} {(p[1] / tile_height):.6f}" for p in coords[:-1]])
                        lf.write(f"{class_id} {coords_str}\n")

def _shift_polygon(polygon, shift_x, shift_y):
    """
    Shift the polygon's coordinates by subtracting shift_x and shift_y.
    """
    return translate(polygon, xoff=-shift_x, yoff=-shift_y)

if __name__ == "__main__":
    # Example usage:
    input_image = "test_dataset/images/1.png"
    input_label = "test_dataset/labels/1.txt"
    output_directory = "test_dataset/output"

    # Split into smaller tiles (640x640 in this example)
    split_image_and_annotations(
        image_path=input_image,
        label_path=input_label,
        output_dir=output_directory,
        tile_width=640,
        tile_height=640
    )
