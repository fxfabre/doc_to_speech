import posixpath

import cv2
from pathlib import Path
from ocr_api.src.img_processing import remove_noise


def main():
    for file_path in Path("images").glob("raw_*"):
        print("Processing", file_path)
        file_name = file_path.name[4:]

        gray_file_path = posixpath.join("images_tmp", "gray_" + file_name)
        clean_file_path = posixpath.join("images", "clean_" + file_name)

        # Charger l'image en couleur
        rgb_img = cv2.imread(file_path.as_posix())
        print("image.shape", rgb_img.shape)

        # Convertir en niveaux de gris
        gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(gray_file_path, gray_img)

        clean_img = remove_noise(gray_img, 41, 5)
        cv2.imwrite(clean_file_path, clean_img)


def sandbox():
    gray_img = cv2.imread("images_tmp/gray_image_1.jpg", cv2.IMREAD_GRAYSCALE)

    # Todo : pivoter image + Tesseract pour différencier zones avec texte vs zones bruit


if __name__ == '__main__':
    main()
