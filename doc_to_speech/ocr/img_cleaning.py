from pathlib import Path

import cv2

from doc_to_speech.ocr.img_processing import remove_noise


def clean_image(file_path: Path) -> str:
    print("Processing", file_path)

    gray_file_path = Path("images_tmp") / ("gray_" + file_path.name)
    clean_file_path = Path("images_tmp") / ("clean_" + file_path.name)

    # Charger l'image en couleur
    rgb_img = cv2.imread(file_path.as_posix())
    print("image.shape", rgb_img.shape)

    # Convertir en niveaux de gris
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    save_picture("gray file", gray_file_path, gray_img)

    clean_img = remove_noise(gray_img, 41, 5)
    save_picture("clean file", clean_file_path, clean_img)
    return clean_file_path.as_posix()


def save_picture(image_name: str, file_path: Path, image: cv2.typing.MatLike) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Saving image {image_name} at", file_path)
    is_success = cv2.imwrite(file_path.as_posix(), image)
    if not is_success:
        raise Exception(f"Failed to save image {image_name}")
