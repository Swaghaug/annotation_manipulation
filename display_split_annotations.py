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
        
        # Load the saved annotated image
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
                    if len(items) < 6:
                        print(f"Skipping annotation with insufficient vertices: {line.strip()}")
                        continue
                    class_id = items[0]
                    coords = list(map(float, items[1:]))
                    points = [(coords[i] * self.original_image.width, coords[i + 1] * self.original_image.height) for i in range(0, len(coords), 2)]
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
                for line in f.readlines():
                    items = line.strip().split()
                    if len(items) < 6:  # Skip invalid annotations
                        continue
                    class_id = items[0]
                    coords = list(map(float, items[1:]))
                    points = [(int(coords[i] * 640), int(coords[i + 1] * 640)) for i in range(0, len(coords), 2)]
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
            cv2.putText(tile_image, class_id, points[0], fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255), thickness=1)

        # Ensure the tile image matches the height of the annotated original
        original_height, original_width, _ = self.annotated_original.shape
        tile_height, tile_width, _ = tile_image.shape

        if tile_height != original_height:
            tile_image = cv2.resize(tile_image, (tile_width, original_height))

        # Combine original and tile images side-by-side
        combined_image = cv2.hconcat([self.annotated_original, tile_image])

        # Resize the combined image to fit in a 1920x1080 window
        window_width, window_height = 1920, 1080
        combined_height, combined_width, _ = combined_image.shape
        scaling_factor = min(window_width / combined_width, window_height / combined_height)
        resized_image = cv2.resize(combined_image, (int(combined_width * scaling_factor), int(combined_height * scaling_factor)))

        # Display the resized image
        cv2.imshow("Tile Viewer", resized_image)



    def start(self):
        while True:
            self.display_tile()
            key = cv2.waitKey(0)

            if key == 27:  # ESC to quit
                print("Exiting...")
                break
            elif key == ord("d"):  # Right arrow or 'd' for next tile
                self.current_tile_index = (self.current_tile_index + 1) % len(self.tile_images)
            elif key == ord("a"):  # Left arrow or 'a' for previous tile
                self.current_tile_index = (self.current_tile_index - 1) % len(self.tile_images)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    original_image_path = "test_dataset/images/1.png"
    output_directory = "test_dataset/output"

    viewer = TileViewer(original_image_path, output_directory)
    viewer.start()
