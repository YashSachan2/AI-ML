from flask import Flask, render_template, request, jsonify
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Load the BLIP model and processor from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Ensure the static folder exists
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Function to generate image description
def generate_description(image):
    # Preprocess the image for the model
    inputs = processor(images=image, return_tensors="pt")
    
    # Generate the description
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    
    return description

# Route for uploading image and generating description
@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        # Get the uploaded file
        file = request.files["image"]
        
        # Save the file to the static/uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Open the image using PIL
        image = Image.open(file_path)
        
        # Generate description
        description = generate_description(image)
        
        # Return the description and image file path
        return render_template("index.html", description=description, image_file=file.filename)
    
    return render_template("index.html", description=None)

if __name__ == "__main__":
    app.run(debug=True)
