import os
from PIL import Image, ImageDraw
import cv2
import numpy as np

class TileViewer:
    def __init__(self, original_image_path, output_dir):
        self.original_image_path = original_image_path
        self.original_label_path = original_image_path.replace("images", "labels").replace(".png", ".txt")
        self.tile_image_dir = os.path.join(output_dir, "images")
        self.tile_label_dir = os.path.join(output_dir, "labels")
        self.annotated_original_path = os.path.join(output_dir, "annotated_original.jpg")
        
        # Load original image
        self.original_image = Image.open(original_image_path)
        
        # Annotate and save the original image with annotations (always overwrite)
        self.create_annotated_original_image()
        
        # Load the saved annotated image via OpenCV (so we can work in OpenCV space)
        self.annotated_original = cv2.imread(self.annotated_original_path)
        
        # Get sorted list of tiles
        self.tile_images = sorted([f for f in os.listdir(self.tile_image_dir) if f.endswith(".jpg")])
        if not self.tile_images:
            raise ValueError(f"No tile images found in {self.tile_image_dir}")
        
        # Current tile index
        self.current_tile_index = 0

    def create_annotated_original_image(self):
        """
        Creates and saves the original image with all annotations, replacing any existing file.
        """
        annotated_image = self.original_image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        try:
            with open(self.original_label_path, "r") as f:
                for line in f.readlines():
                    items = line.strip().split()
                    # Expect class_id + at least 2 pairs of coordinates (i.e., 6 items total)
                    if len(items) < 6:
                        print(f"Skipping annotation with insufficient vertices: {line.strip()}")
                        continue
                    class_id = items[0]
                    coords = list(map(float, items[1:]))

                    # Assume normalized coordinates in [0,1], multiply by image dimensions
                    points = [
                        (coords[i] * self.original_image.width, coords[i + 1] * self.original_image.height)
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
        """
        Loads the tile annotations. Adjust this if your tile annotation format differs.
        Expects: class_id x1 y1 x2 y2 ...
        with normalized coordinates in [0,1] for a 640×640 tile.
        """
        annotations = []
        try:
            with open(label_path, "r") as f:
                for line in f.readlines():
                    items = line.strip().split()
                    if len(items) < 6:  # Skip invalid annotations
                        continue
                    class_id = items[0]
                    coords = list(map(float, items[1:]))

                    # Convert normalized coords to pixel coords for a 640×640 tile
                    points = [
                        (int(coords[i] * 640), int(coords[i + 1] * 640))
                        for i in range(0, len(coords), 2)
                    ]
                    annotations.append((class_id, points))
        except FileNotFoundError:
            pass
        return annotations

    def display_tile(self):
        tile_name = self.tile_images[self.current_tile_index]
        tile_path = os.path.join(self.tile_image_dir, tile_name)
        label_path = os.path.join(self.tile_label_dir, tile_name.replace(".jpg", ".txt"))
        
        # Load tile image and annotations
        tile_image = cv2.imread(tile_path)
        tile_annotations = self.load_annotations(label_path)

        # Annotate tile
        for class_id, points in tile_annotations:
            points = np.array(points, dtype=np.int32)
            cv2.polylines(tile_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
            for x, y in points:
                cv2.circle(tile_image, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
            cv2.putText(tile_image, class_id, points[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(255, 255, 255), thickness=1)

        # Get sizes
        orig_height, orig_width, _ = self.annotated_original.shape
        tile_height, tile_width, _ = tile_image.shape

        # 1) Make both images the same display height to avoid tall vs. short mismatch.
        #    Preserve the aspect ratio for the tile image.
        if tile_height != orig_height:
            scale_factor = orig_height / float(tile_height)
            new_tile_width = int(tile_width * scale_factor)
            new_tile_height = orig_height
            tile_image = cv2.resize(tile_image, (new_tile_width, new_tile_height))

        # 2) Combine original and tile images side-by-side
        combined_image = cv2.hconcat([self.annotated_original, tile_image])
        
        # 3) Resize the combined image if it exceeds 1920×1080
        window_width, window_height = 1920, 1080
        combined_height, combined_width, _ = combined_image.shape
        if combined_width > window_width or combined_height > window_height:
            scaling_factor = min(window_width / combined_width, window_height / combined_height)
            new_width = int(combined_width * scaling_factor)
            new_height = int(combined_height * scaling_factor)
            combined_image = cv2.resize(combined_image, (new_width, new_height))

        # Display the final image
        cv2.imshow("Tile Viewer", combined_image)

    def start(self):
        while True:
            self.display_tile()
            key = cv2.waitKey(0)

            if key == 27:  # ESC to quit
                print("Exiting...")
                break
            elif key == ord("d"):  # 'd' for next tile
                self.current_tile_index = (self.current_tile_index + 1) % len(self.tile_images)
            elif key == ord("a"):  # 'a' for previous tile
                self.current_tile_index = (self.current_tile_index - 1) % len(self.tile_images)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    original_image_path = "test_dataset/images/1.png"
    output_directory = "test_dataset/output"

    viewer = TileViewer(original_image_path, output_directory)
    viewer.start()
