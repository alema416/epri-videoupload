from edgetpumodel EdgeTPUModel
from utils import get_image_tensor

model = EdgeTPUModel("/yolo/edgetpu-yolo/epri-videoupload/best-int8.tflite", "/yolo/edgetpu-yolo/epri-videoupload/custom_epri.yaml")
input_shape = model.get_input_shape()

full_image, net_image, pad = get_image_tensor("/yolo/edgetpu-yolo/epri-videoupload/101.JPG", input_shape[0])
pred = model.predict(net_image)