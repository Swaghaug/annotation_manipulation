import os
import uuid
import zipfile
import shutil

from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
from splitter import split_images_in_folder  # your script's function

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """
    Handles both images.zip and labels.zip from the form.
    We store them, unzip them, run tile logic, then let the front-end know it's done.
    """
    # 1) Grab the two files
    images_file = request.files.get("images_zip")
    labels_file = request.files.get("labels_zip")
    if not images_file or not labels_file:
        return "Missing one or both zip files", 400
    
    # 2) Create a unique batch ID
    batch_id = str(uuid.uuid4())[:8]
    batch_dir = os.path.join(UPLOAD_FOLDER, batch_id)
    os.makedirs(batch_dir, exist_ok=True)
    
    # 3) Save the uploaded zips
    images_zip_path = os.path.join(batch_dir, images_file.filename)
    labels_zip_path = os.path.join(batch_dir, labels_file.filename)
    images_file.save(images_zip_path)
    labels_file.save(labels_zip_path)
    
    # 4) Extract both zips into batch_dir
    with zipfile.ZipFile(images_zip_path, "r") as z:
        z.extractall(batch_dir)
    with zipfile.ZipFile(labels_zip_path, "r") as z:
        z.extractall(batch_dir)
    
    # We assume the extracted structure is now batch_dir/images and batch_dir/labels
    # 5) Perform splitting
    output_path = os.path.join(OUTPUT_FOLDER, batch_id)
    os.makedirs(output_path, exist_ok=True)
    
    split_images_in_folder(
        input_dir=batch_dir,       # contains images/ and labels/
        output_dir=output_path,    # results go here
        tile_width=640,            # adjust if needed
        tile_height=640
    )
    
    # 6) Return a JSON response with the batch_id so we can enable the UI
    return jsonify({"status": "ok", "batch_id": batch_id})

@app.route("/download/<batch_id>", methods=["GET"])
def download(batch_id):
    """
    Zips up the processed dataset and returns it.
    """
    output_dir = os.path.join(OUTPUT_FOLDER, batch_id)
    if not os.path.exists(output_dir):
        return "No such dataset found.", 404
    
    zip_name = f"{batch_id}_tiles.zip"
    zip_path = os.path.join(OUTPUT_FOLDER, zip_name)
    
    # If it already exists, remove
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    shutil.make_archive(
        base_name=zip_path.replace(".zip", ""),
        format="zip",
        root_dir=output_dir
    )
    
    return send_file(zip_path, as_attachment=True)

@app.route("/display/<batch_id>")
def display_results(batch_id):
    """
    A simple page that shows the splitted tile images in a listing or small gallery.
    """
    output_dir = os.path.join(OUTPUT_FOLDER, batch_id)
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        return "No images found for this dataset", 404
    
    # List tile images
    tile_files = sorted(os.listdir(images_dir))
    tile_files = [f for f in tile_files if f.lower().endswith(".jpg")]
    
    return render_template("results.html", batch_id=batch_id, tiles=tile_files)

if __name__ == "__main__":
    app.run(debug=True)
