import os
import cv2

BASE_PATH = "data"

for img_name in os.listdir(BASE_PATH):
    if not img_name.endswith(".jpg"):
        continue
    img_path = os.path.join(BASE_PATH, img_name)
    img = cv2.imread(img_path)
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(BASE_PATH, img_name), img_grayscale)
    
    print(f"converted {img_name} to grayscale")
    

    