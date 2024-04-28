import cv2
import numpy as np


def gen_images():
    # Charger l'image en couleur
    image = cv2.imread("images/page.png")
    print("image.shape", image.shape)

    # Convertir en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("images_tmp/gray_image.jpg", gray_image)

    # # Appliquer une équilibration d'histogramme pour améliorer le contraste
    # equalized_image = cv2.equalizeHist(gray_image)
    # cv2.imwrite("images_tmp/equalized_image.jpg", equalized_image)

    # Appliquer un flou gaussien pour réduire le bruit
    blurred_image_3 = cv2.GaussianBlur(gray_image, (3, 3), 0)
    cv2.imwrite("images_tmp/blurred_image_3.jpg", blurred_image_3)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    cv2.imwrite("images_tmp/blurred_image_5.jpg", blurred_image)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    cv2.imwrite("images_tmp/blurred_image_7.jpg", blurred_image)

    return gray_image


def seuillage():
    rgb_image = cv2.imread("images_tmp/gray_image.jpg")
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    print("image.shape", gray_image.shape)

    # Appliquer le seuillage binaire
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("images_tmp/thresh.jpg", thresh)

    # Seuillage adaptatif
    thresh_adaptive = cv2.adaptiveThreshold(
        thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 2
    )
    cv2.imwrite("images_tmp/thresh_adaptive.jpg", thresh_adaptive)

    # Seuillage avec la méthode de Otsu
    ret2, thresh_otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite("images_tmp/thresh_otsu.jpg", thresh_otsu)


def rotation_image(angle_in_degrees=None):
    gray_image = cv2.imread("images_tmp/equalized_image.jpg")

    if angle_in_degrees is None:
        # Calculer les contours pour déterminer l'angle de rotation
        edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        angle = np.mean([line[0][1] for line in lines])  # Calcule l'angle moyen
        angle_in_degrees = np.rad2deg(angle) - 90  # Convertit en degrés et ajuste

    # Appliquer la rotation
    if abs(angle_in_degrees) >= 2:
        print("Rotation :", angle_in_degrees)
        (h, w) = gray_image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle_in_degrees, 1.0)
        rotated_image = cv2.warpAffine(gray_image, M, (w, h))
    else:
        rotated_image = gray_image

    # Enregistrer l'image pivotée
    cv2.imwrite(f"images_tmp/rotated_image_{int(angle_in_degrees)}.jpg", rotated_image)
    return rotated_image


def detection_contours():
    image = cv2.imread("images/page.png")
    blurred = get_gray_and_blur()

    # Appliquer le seuillage pour obtenir une image binaire
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours sur l'image originale pour visualisation
    contour_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 3)

    # Afficher l'image avec les contours
    cv2.imshow("Contours", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def projection_horizontale():
    # Charger l'image en niveaux de gris
    image = cv2.imread("images/gray_image.jpg", cv2.IMREAD_GRAYSCALE)

    # Appliquer le seuillage binaire
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

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
        cv2.rectangle(image, (0, start), (image.shape[1], end), (255, 0, 0), 2)

    cv2.imshow("Detected Lines", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def boites_englobantes():
    # Charger l'image en niveaux de gris
    image = cv2.imread("images/_blurred_image.jpg", cv2.IMREAD_GRAYSCALE)

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


gen_images()
seuillage()

# rotation_image(-5)
# rotation_image(-10)

# text_cleaning()
# amelioration_contraste()
# suppression_bruit()
# deroration_image()
# boites_englobantes()
