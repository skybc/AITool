from ultralytics import YOLO

# Load a pretrained segmentation model like YOLO11n-seg
model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# Train the model on the Carparts Segmentation dataset
results = model.train(data="carparts-seg.yaml", epochs=100, imgsz=640)

# After training, you can validate the model's performance on the validation set
results = model.val()

# Or perform prediction on new images or videos
results = model.predict("path/to/your/image.jpg")