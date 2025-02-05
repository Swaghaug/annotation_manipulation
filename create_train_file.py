import os

def generate_train_txt(directory):
    images_dir = os.path.join(directory, 'images')
    labels_dir = os.path.join(directory, 'labels')
    output_file = os.path.join(directory, 'Train.txt')
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("Error: 'images' or 'labels' directory does not exist.")
        return
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    label_extensions = {'.txt', '.xml', '.json'}  # Adjust based on label format
    
    image_files = {os.path.splitext(f)[0] for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in image_extensions}
    label_files = {os.path.splitext(f)[0] for f in os.listdir(labels_dir) if os.path.splitext(f)[1].lower() in label_extensions}
    
    common_files = sorted(image_files & label_files)
    
    with open(output_file, 'w') as f:
        for file in common_files:
            f.write(file + '\n')
    
    print(f"Train.txt created with {len(common_files)} entries at {output_file}")

if __name__ == "__main__":
    directory = input("Enter the path to the dataset directory: ").strip()
    generate_train_txt(directory)
