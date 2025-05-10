from ultralytics import YOLO
#load the pre-trained model
model = YOLO("best(2).pt")

#run inference on the source
results = model('C:\\Users\\srika\\OneDrive\\Desktop\\D17\\antalya.jpg', save=True, conf=0.25, show=True)

num_trees = len(results[0].boxes)
print(f"Number of trees detected: {num_trees}")

