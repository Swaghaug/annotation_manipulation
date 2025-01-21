import os
import argparse
from PIL import Image, ImageDraw
import cv2
import numpy as np

class TileViewer:
    def __init__(
        self,
        original_image_path,
        tiles_dir,
        show_all_tiles=False,
        tile_width=640,
        tile_height=640
    ):
        self.original_image_path = original_image_path
        self.tiles_dir = tiles_dir
        self.show_all_tiles = show_all_tiles
        self.tile_width = tile_width
        self.tile_height = tile_height

        self.image_base = os.path.splitext(os.path.basename(original_image_path))[0]
        self.original_label_path = (
            original_image_path
            .replace("images", "labels")
            .replace(".png", ".txt")
            .replace(".jpg", ".txt")
        )

        self.tile_image_dir = os.path.join(tiles_dir, "images")
        self.tile_label_dir = os.path.join(tiles_dir, "labels")

        self.annotated_original_path = os.path.join(tiles_dir, f"annotated_{self.image_base}.jpg")

        # Load original image with PIL to get dimensions
        self.original_image = Image.open(original_image_path)
        self.full_width, self.full_height = self.original_image.size

        # Create and save the annotated original
        self.create_annotated_original_image()

        # Load via OpenCV for display
        self.annotated_original = cv2.imread(self.annotated_original_path)
        if self.annotated_original is None:
            raise ValueError(f"Failed to load annotated image from {self.annotated_original_path}")

        # Gather tiles matching the base name
        all_tile_images = sorted(
            f for f in os.listdir(self.tile_image_dir)
            if f.endswith(".jpg") and f.startswith(self.image_base + "_")
        )
        if not all_tile_images:
            raise ValueError(f"No tile images found for {self.image_base} in {self.tile_image_dir}")

        # Optionally filter out empty-annotation tiles
        self.tile_images = []
        for tile_name in all_tile_images:
            label_path = os.path.join(self.tile_label_dir, tile_name.replace(".jpg", ".txt"))
            tile_annotations = self.load_annotations(label_path)
            if tile_annotations or self.show_all_tiles:
                self.tile_images.append(tile_name)

        if not self.tile_images:
            raise ValueError(
                f"All tiles for {self.image_base} have zero annotations, "
                f"and --show_all_tiles=False, so nothing to display."
            )

        self.current_tile_index = 0

    def create_annotated_original_image(self):
        annotated_image = self.original_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        try:
            with open(self.original_label_path, "r") as f:
                for line in f:
                    items = line.strip().split()
                    if len(items) < 6:
                        print(f"Skipping annotation with insufficient vertices: {line.strip()}")
                        continue
                    class_id = items[0]
                    coords = list(map(float, items[1:]))

                    width, height = annotated_image.size
                    points = [
                        (coords[i] * width, coords[i + 1] * height)
                        for i in range(0, len(coords), 2)
                    ]
                    draw.polygon(points, outline="green", width=2)
                    for x, y in points:
                        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
                    draw.text(points[0], class_id, fill="blue")
        except FileNotFoundError:
            print(f"No annotation file found at {self.original_label_path}. Creating plain original image.")

        annotated_image.save(self.annotated_original_path)
        print(f"Saved annotated original image to {self.annotated_original_path}")

    def load_annotations(self, label_path):
        annotations = []
        try:
            with open(label_path, "r") as f:
                for line in f:
                    items = line.strip().split()
                    if len(items) < 6:
                        continue
                    class_id = items[0]
                    coords = list(map(float, items[1:]))

                    points = [
                        (int(coords[i] * self.tile_width),
                         int(coords[i + 1] * self.tile_height))
                        for i in range(0, len(coords), 2)
                    ]
                    annotations.append((class_id, points))
        except FileNotFoundError:
            pass
        return annotations

    def parse_tile_row_col(self, tile_name):
        base = os.path.splitext(tile_name)[0]
        parts = base.split("_")
        try:
            row_idx = int(parts[-2])
            col_idx = int(parts[-1])
        except (IndexError, ValueError):
            row_idx, col_idx = 0, 0
        return row_idx, col_idx

    def display_tile(self):
        tile_name = self.tile_images[self.current_tile_index]
        tile_path = os.path.join(self.tile_image_dir, tile_name)
        label_path = os.path.join(self.tile_label_dir, tile_name.replace(".jpg", ".txt"))

        tile_image = cv2.imread(tile_path)
        tile_annotations = self.load_annotations(label_path)

        # Annotate the tile
        for class_id, points in tile_annotations:
            pts_np = np.array(points, dtype=np.int32)
            cv2.polylines(tile_image, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)
            for (x, y) in points:
                cv2.circle(tile_image, (x, y), 3, (0, 0, 255), -1)
            cv2.putText(
                tile_image,
                class_id,
                (pts_np[0][0], pts_np[0][1]),  # (x, y) tuple
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                thickness=1
            )

        ann_orig_copy = self.annotated_original.copy()

        print(f"Displaying tile grid for {tile_name}")
        row_idx, col_idx = self.parse_tile_row_col(tile_name)

        tile_left = col_idx * self.tile_width
        tile_top = row_idx * self.tile_height
        tile_right = min(tile_left + self.tile_width, self.full_width)
        tile_bottom = min(tile_top + self.tile_height, self.full_height)

        # Scale line thickness as percentage of the smaller dimension
        thickness = max(1, int(0.005 * min(self.full_width, self.full_height)))

        cv2.rectangle(
            ann_orig_copy,
            (tile_left, tile_top),
            (tile_right, tile_bottom),
            (0, 255, 0),
            thickness
        )

        # Resize tile image to match the annotated original height
        orig_h, orig_w, _ = ann_orig_copy.shape
        tile_h, tile_w, _ = tile_image.shape

        if tile_h != orig_h:
            scale_factor = orig_h / float(tile_h)
            new_tile_w = int(tile_w * scale_factor)
            new_tile_h = orig_h
            tile_image = cv2.resize(tile_image, (new_tile_w, new_tile_h))

        # Combine horizontally
        combined = cv2.hconcat([ann_orig_copy, tile_image])

        # Scale down if bigger than 1920×1080
        window_w, window_h = 1920, 1080
        ch, cw, _ = combined.shape
        if cw > window_w or ch > window_h:
            sf = min(window_w / cw, window_h / ch)
            new_w = int(cw * sf)
            new_h = int(ch * sf)
            combined = cv2.resize(combined, (new_w, new_h))

        cv2.imshow("Tile Viewer - Press esc to go to next image", combined)

    def start(self):
        while True:
            self.display_tile()
            key = cv2.waitKey(0)
            
            if key in [ord('d'), 83]:  # 'd' or right arrow
                self.current_tile_index = (self.current_tile_index + 1) % len(self.tile_images)
            elif key in [ord('a'), 81]:  # 'a' or left arrow
                self.current_tile_index = (self.current_tile_index - 1) % len(self.tile_images)
            elif key == 27:  # ESC
                print("Exiting this image...")
                break

        cv2.destroyAllWindows()


def annotate_folder(
    original_images_dir,
    tiles_dir,
    show_all_tiles=False,
    tile_width=640,
    tile_height=640
):
    original_images_path = os.path.join(original_images_dir, "images")
    valid_exts = (".png", ".jpg", ".jpeg")
    all_images = [f for f in os.listdir(original_images_path) if f.lower().endswith(valid_exts)]
    if not all_images:
        print(f"No images found in directory: {original_images_path}")
        return

    all_images.sort()

    for img_file in all_images:
        image_path = os.path.join(original_images_path, img_file)
        print(f"\n--- Now annotating: {image_path} ---")
        try:
            viewer = TileViewer(
                original_image_path=image_path,
                tiles_dir=tiles_dir,
                show_all_tiles=show_all_tiles,
                tile_width=tile_width,
                tile_height=tile_height
            )
            viewer.start()
        except ValueError as e:
            print(f"Skipping {img_file} because: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View original images with annotated tiles side-by-side.")
    parser.add_argument(
        "--original_images_dir",
        type=str,
        required=True,
        help="Base directory with 'images' subfolder for original images (e.g. 'batch3')."
    )
    parser.add_argument(
        "--tiles_dir",
        type=str,
        required=True,
        help="Directory containing tile images and labels (e.g. 'test_dataset/output')."
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
    parser.add_argument(
        "--show_all_tiles",
        action="store_true",
        default=False,
        help="If set, display tiles even if they have zero annotations."
    )

    args = parser.parse_args()

    annotate_folder(
        original_images_dir=args.original_images_dir,
        tiles_dir=args.tiles_dir,
        show_all_tiles=args.show_all_tiles,
        tile_width=args.tile_width,
        tile_height=args.tile_height
    )
