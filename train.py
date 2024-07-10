from ultralytics import YOLO

# Load a model
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("runs/detect/train3/best.pt")  # load a pretrained model (recommended for training)

# # Use the model
model.train(data="datasets/fav/dataset.yaml", epochs=300)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format
