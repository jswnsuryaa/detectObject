import cv2
import numpy as np
import os

# Tentukan jalur absolut ke folder yang berisi file
base_path = 'C:/Users/joswa/OneDrive/Desktop/New folder/ppk'

# Tentukan jalur absolut ke file cfg, weights, dan coco.names
cfg_path = os.path.join(base_path, 'yolo.cfg')
weights_path = os.path.join(base_path, 'yolov3.weights')
names_path = os.path.join(base_path, 'coco.names')
image_path = os.path.join(base_path, 'kota.jpg')  # Asumsi gambar input ada dengan nama input.jpg

# Periksa apakah file benar-benar ada
assert os.path.exists(cfg_path), f"File not found: {cfg_path}"
assert os.path.exists(weights_path), f"File not found: {weights_path}"
assert os.path.exists(names_path), f"File not found: {names_path}"
assert os.path.exists(image_path), f"File not found: {image_path}"

# Muat model YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Muat kelas COCO
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Baca gambar dari file
image = cv2.imread(image_path)
height, width, channels = image.shape

# Siapkan gambar untuk YOLO
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)

# Dapatkan deteksi dari YOLO
outs = net.forward(output_layers)

# Inisialisasi daftar untuk bounding box, confidence, dan class ID
boxes = []
confidences = []
class_ids = []

# Inisialisasi kamus untuk menghitung jumlah objek per kelas
object_counts = {class_name: 0 for class_name in classes}

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            object_counts[classes[class_id]] += 1  # Tambahkan jumlah objek terdeteksi

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

detected = False
for i in range(len(boxes)):
    if i in indexes:
        detected = True
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (0, 255, 0)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

# Tentukan nama file output yang unik
output_image_index = 1
output_image_path = os.path.join(base_path, f'output{output_image_index}.jpg')
while os.path.exists(output_image_path):
    output_image_index += 1
    output_image_path = os.path.join(base_path, f'output{output_image_index}.jpg')

# Simpan gambar output
cv2.imwrite(output_image_path, image)

# Tampilkan gambar output
cv2.imshow('Object Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Tampilkan status sukses pendeteksian
if detected:
    print("Deteksi objek berhasil!")
else:
    print("Tidak ada objek yang terdeteksi.")

# Tampilkan jumlah objek yang terdeteksi untuk setiap kelas
print("Jumlah objek yang terdeteksi:")
for class_name, count in object_counts.items():
    if count > 0:
        print(f"{class_name}: {count}")
