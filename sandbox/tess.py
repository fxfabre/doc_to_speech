import cv2
import pandas as pd
import pytesseract
from pytesseract import Output

# Configuration de Tesseract
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'


def rotate_image(image, angle):
    """ Rotate the image by the given angle """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


# Chargement de l'image
image = cv2.imread("images/raw_page_1.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Détecter le texte
d = pytesseract.image_to_data(gray, output_type=Output.DICT)
df_txt_data = pd.DataFrame(d)

print()


from pprint import pprint
pprint(d, width=200)

for idx, shape_info in df_txt_data.iterrows():
    if shape_info["text"].strip() == "":
        continue

    print(shape_info)
    x = shape_info["left"]
    y = shape_info["top"]
    w = shape_info["width"]
    h = shape_info["height"]
    cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    cv2.imshow("Tesseract txt", image)
    cv2.waitKey(0)

print("end")
cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()

exit(0)

# Trouver l'angle de rotation basé sur les positions de texte
angles = []
for i in range(len(d['text'])):
    if int(d['conf'][i]) > 60:  # seulement les confiances élevées
        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        # L'angle est calculé par atan2(y, x)
        angles.append(np.arctan2(h, w))

# Calculer l'angle moyen
median_angle = np.median(angles) * (180 / np.pi)

# Rotate the original image by the negative of the calculated angle
corrected_image = rotate_image(image, -median_angle)

cv2.imshow('Original', image)
cv2.imshow('Corrected', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
