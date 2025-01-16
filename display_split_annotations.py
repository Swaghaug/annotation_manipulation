import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_polygon_annotations(annotation_path):
    polygons = []
    try:
        with open(annotation_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                label = parts[0]
                points = np.array(parts[1:], dtype=float).reshape(-1, 2)  # Convert to pairs of (x, y)
                polygons.append((label, points))
    except FileNotFoundError:
        print(f"Annotation file not found: {annotation_path}")
    return polygons

def annotate_image_with_polygons(image, annotations, image_size):
    h, w, _ = image.shape
    annotated_image = image.copy()
    for label, points in annotations:
        # Scale points to image size
        scaled_points = (points * [w, h]).astype(int)
        
        # Draw the polygon lines in green
        cv2.polylines(annotated_image, [scaled_points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Draw the vertex points in red
        for x, y in scaled_points:
            cv2.circle(annotated_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        
        # Draw the label near the first point
        if len(scaled_points) > 0:
            x, y = scaled_points[0]
            cv2.putText(annotated_image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return annotated_image

def display_images(original_image, original_annotations, tile_image, tile_annotations, original_size):
    # Annotate the original image with polygons
    annotated_original = annotate_image_with_polygons(original_image, original_annotations, original_size)
    
    # Annotate the tile image with polygons
    annotated_tile = annotate_image_with_polygons(tile_image, tile_annotations, original_size)
    
    # Display the images
    plt.figure(figsize=(15, 7))
    
    # Original image with annotations
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(annotated_original, cv2.COLOR_BGR2RGB))
    plt.title("Original Image with Polygon Annotations")
    plt.axis("off")
    
    # First tile with annotations
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(annotated_tile, cv2.COLOR_BGR2RGB))
    plt.title("Tile (0,0) with Polygon Annotations")
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    # Paths
    original_image_path = "test_dataset/images/1.png"
    original_annotation_path = "test_dataset/labels/1.txt"
    tile_image_path = "test_dataset/output/images/1_0_0.jpg"
    tile_annotation_path = "test_dataset/output/labels/1_0_0.txt"
    
    # Image sizes
    original_size = (1792, 1024)  # Original image size
    tile_size = (640, 640)  # Tile size
    
    # Load images
    original_image = cv2.imread(original_image_path)
    tile_image = cv2.imread(tile_image_path)
    
    if original_image is None:
        print("Error: Original image not found.")
        return
    if tile_image is None:
        print("Error: Tile image not found.")
        return
    
    # Load polygon annotations
    original_annotations = load_polygon_annotations(original_annotation_path)
    tile_annotations = load_polygon_annotations(tile_annotation_path)
    
    # Display annotated images
    display_images(original_image, original_annotations, tile_image, tile_annotations, original_size)

if __name__ == "__main__":
    main()
