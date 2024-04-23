import cv2
import numpy as np


def text_cleaning():
    # Charger l'image en couleur
    image = cv2.imread("images/page.png")

    # Convertir en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Enregistrer l'image en niveaux de gris
    cv2.imwrite("images/_gray_image.jpg", gray_image)


def amelioration_contraste():
    # Charger l'image en niveaux de gris
    gray_image = cv2.imread("images/_gray_image.jpg", cv2.IMREAD_GRAYSCALE)

    # Appliquer une équilibration d'histogramme pour améliorer le contraste
    equalized_image = cv2.equalizeHist(gray_image)

    # Enregistrer l'image avec contraste amélioré
    cv2.imwrite("images/_equalized_image.jpg", equalized_image)


def suppression_bruit():
    # Charger l'image en niveaux de gris
    gray_image = cv2.imread("images/_gray_image.jpg", cv2.IMREAD_GRAYSCALE)

    # Appliquer un flou gaussien pour réduire le bruit
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Enregistrer l'image floutée
    cv2.imwrite("images/_blurred_image.jpg", blurred_image)


def deroration_image():
    """rotation de l'image"""

    # Charger l'image
    image = cv2.imread("images/_gray_image.jpg")

    # Calculer les contours pour déterminer l'angle de rotation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    angle = np.mean([line[0][1] for line in lines])  # Calcule l'angle moyen
    angle_in_degrees = np.rad2deg(angle) - 90  # Convertit en degrés et ajuste

    # Appliquer la rotation
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_in_degrees, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h))

    # Enregistrer l'image dérotée
    cv2.imwrite("images/_rotated_image.jpg", rotated_image)


def detection_contours():
    # Charger l'image en niveaux de gris
    image = cv2.imread("images/_gray_image.jpg", cv2.IMREAD_GRAYSCALE)

    # Appliquer un flou pour réduire le bruit
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

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
    image = cv2.imread("images/_gray_image.jpg", cv2.IMREAD_GRAYSCALE)

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


# text_cleaning()
# amelioration_contraste()
# suppression_bruit()
# deroration_image()
boites_englobantes()
