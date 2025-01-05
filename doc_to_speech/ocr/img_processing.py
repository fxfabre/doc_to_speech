import cv2
import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from matplotlib import pyplot as plt


def extract_text_from_image(image_path: str) -> str:
    model = ocr_predictor(
        det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
    )
    doc = DocumentFile.from_images(image_path)
    result = model(doc)

    lines = []
    for page_json in sorted(
        result.export()["pages"], key=lambda page: page["page_idx"]
    ):
        print(
            "page",
            page_json["page_idx"],
            "size",
            page_json["dimensions"],
            page_json["language"],
        )

        for bloc in page_json["blocks"]:
            for line in bloc["lines"]:
                lines.append(" ".join(w["value"] for w in line["words"]))

    return " ".join(lines)


####################
# Fonctions de test gardées pour référence
####################


def erode(image: cv2.typing.MatLike, erosion_size=4) -> cv2.typing.MatLike:
    """
    Erosion d'une image

    :param image:
    :param erosion_size:
    :return:
    """
    element = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (2 * erosion_size + 1, 2 * erosion_size + 1),
        (erosion_size, erosion_size),
    )
    return cv2.erode(image, element)


def remove_noise(
    gray_img: cv2.typing.MatLike, block_size=41, erosion_size=5
) -> cv2.typing.MatLike:
    """
    Le seuillage local :
    - Permet de mieux filtrer les zones sombres (bords de l'image)
    - Mais fait ressortir des caractères par transparence (de la page suivante) dans les zones claires

    On rajoute un mask dessus pour filtrer les zones avec les pixels les plus intenses
    """

    # Seuillage local de l'image
    thresh_img = cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2
    )

    # Erosion de l'image, pour définir le mask des pixels à garder
    eroded_img = erode(gray_img, erosion_size=erosion_size)
    # Seuillage pour restreindre le mask des pixels à garder
    _, mask = cv2.threshold(eroded_img, 50, 255, cv2.THRESH_BINARY_INV)

    # Application du mask
    return cv2.bitwise_or(thresh_img, 255 - mask)


def sandbox_filtering():
    """
    Test des différents filtres simples sur une image :
    https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
    """

    gray_img = cv2.imread("images_tmp/gray_page_1.jpg", cv2.IMREAD_GRAYSCALE)
    assert gray_img is not None, "file could not be read, check with os.path.exists()"

    # 2D convolution
    kernel = np.ones((5, 5), np.float32)
    kernel = kernel / kernel.sum()
    convolution_img = cv2.filter2D(gray_img, -1, kernel)

    # Blur
    # average_blur = cv2.blur(gray_img, (5, 5))
    gauss_blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    median_blur = cv2.medianBlur(gray_img, 5)
    bilateral = cv2.bilateralFilter(gray_img, 7, sigmaColor=100, sigmaSpace=150)

    # Display. 152 = 1 row, 5 cols, index 2
    plt.subplot(151), plt.imshow(gray_img, cmap="gray"), plt.title("Original")
    plt.xticks([]), plt.yticks([])

    (
        plt.subplot(152),
        plt.imshow(convolution_img, cmap="gray"),
        plt.title("2D convolution"),
    )
    plt.xticks([]), plt.yticks([])

    (
        plt.subplot(153),
        plt.imshow(gauss_blur, cmap="gray"),
        plt.title("Gaussian Blurring"),
    )
    plt.xticks([]), plt.yticks([])

    plt.subplot(154), plt.imshow(median_blur, cmap="gray"), plt.title("Median Blurring")
    plt.xticks([]), plt.yticks([])

    (
        plt.subplot(155),
        plt.imshow(bilateral, cmap="gray"),
        plt.title("Bilateral Filtering"),
    )
    plt.xticks([]), plt.yticks([])

    plt.show()


def seuillage():
    gray_image = cv2.imread("images_tmp/gray_image.jpg", flags=cv2.IMREAD_GRAYSCALE)
    print("image.shape", gray_image.shape)

    # Appliquer le seuillage binaire
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("images_tmp/thresh.jpg", thresh)

    # Seuillage adaptatif
    for block_size in [71, 111, 131, 161, 191]:
        thresh_adaptive = cv2.adaptiveThreshold(
            gray_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            2,
        )
        cv2.imwrite(f"images_tmp/thresh_adaptive_{block_size}.jpg", thresh_adaptive)

    # Seuillage avec la méthode de Otsu
    ret2, thresh_otsu = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cv2.imwrite("images_tmp/thresh_otsu.jpg", thresh_otsu)


def rotation_image(angle_in_degrees=None):
    gray_img = cv2.imread("images/gray_image.jpg", cv2.IMREAD_GRAYSCALE)

    if angle_in_degrees is None:
        # Calculer les contours pour déterminer l'angle de rotation
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        angle = np.mean([line[0][1] for line in lines])  # Calcule l'angle moyen
        angle_in_degrees = np.rad2deg(angle) - 90  # Convertit en degrés et ajuste

    # Appliquer la rotation
    if abs(angle_in_degrees) >= 2:
        print("Rotation :", angle_in_degrees)
        (h, w) = gray_img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle_in_degrees, 1.0)
        rotated_image = cv2.warpAffine(gray_img, M, (w, h))
    else:
        rotated_image = gray_img

    # Enregistrer l'image pivotée
    cv2.imwrite(f"images_tmp/rotated_image_{int(angle_in_degrees)}.jpg", rotated_image)
    return rotated_image


def detection_contours():
    rgb_img = cv2.imread("images/raw_page_1.jpg")
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    blur_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

    # Appliquer le seuillage pour obtenir une image binaire
    _, thresh = cv2.threshold(blur_img, 127, 255, cv2.THRESH_BINARY_INV)

    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours sur l'image originale pour visualisation
    contour_image = cv2.drawContours(rgb_img.copy(), contours, -1, (0, 255, 0), 3)

    # Afficher l'image avec les contours
    cv2.imshow("Contours", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def projection_horizontale():
    """Marche pas ?!"""

    # Charger l'image en niveaux de gris
    gray_img = cv2.imread("images_tmp/gray_page_1.png", cv2.IMREAD_GRAYSCALE)

    # Appliquer le seuillage binaire
    thresh = remove_noise(gray_img)

    # Calculer la projection horizontale
    projection = np.sum(thresh, axis=1)

    # Détecter les lignes avec des concentrations élevées de pixels noirs
    lines = []
    threshold = 50  # Seuil pour déterminer si une ligne contient du texte
    in_line = False
    for i, val in enumerate(projection):
        if val > threshold and not in_line:
            # Début d'une nouvelle ligne de texte
            start = i
            in_line = True
        elif val <= threshold and in_line:
            # Fin de la ligne de texte
            end = i
            lines.append((start, end))
            in_line = False

    # Afficher les lignes détectées
    for start, end in lines:
        cv2.rectangle(gray_img, (0, start), (gray_img.shape[1], end), (255, 0, 0), 2)

    cv2.imshow("Detected Lines", gray_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def boites_englobantes():
    # Charger l'image en niveaux de gris
    image = cv2.imread("images_tmp/gray_page_1.jpg", cv2.IMREAD_GRAYSCALE)

    # Appliquer le seuillage binaire
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner des boîtes englobantes autour des contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
