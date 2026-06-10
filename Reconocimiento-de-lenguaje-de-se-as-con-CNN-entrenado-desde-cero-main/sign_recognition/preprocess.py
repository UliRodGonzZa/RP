from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np


ModoBordes = Literal["canny", "sobel"]


@dataclass(frozen=True)
class RangoHSV:
    """Limites HSV usados para separar piel del fondo."""

    inferior: tuple[int, int, int] = (0, 25, 45)
    superior: tuple[int, int, int] = (25, 255, 255)

    @property
    def lower(self) -> tuple[int, int, int]:
        return self.inferior

    @property
    def upper(self) -> tuple[int, int, int]:
        return self.superior


@dataclass(frozen=True)
class ConfiguracionPreprocesamiento:
    """Parametros compartidos por captura, entrenamiento e interfaz."""

    tamano_imagen: int = 64
    rango_hsv: RangoHSV = RangoHSV()
    modo_bordes: ModoBordes = "canny"
    area_minima: int = 1200

    @property
    def image_size(self) -> int:
        return self.tamano_imagen

    @property
    def hsv_range(self) -> RangoHSV:
        return self.rango_hsv

    @property
    def edge_mode(self) -> ModoBordes:
        return self.modo_bordes

    @property
    def min_area(self) -> int:
        return self.area_minima


def limites_roi_central(imagen_bgr: np.ndarray, escala: float = 0.68) -> tuple[int, int, int, int]:
    alto, ancho = imagen_bgr.shape[:2]
    lado = int(min(ancho, alto) * escala)
    x0 = (ancho - lado) // 2
    y0 = (alto - lado) // 2
    return x0, y0, x0 + lado, y0 + lado


def recortar_roi_central(imagen_bgr: np.ndarray, escala: float = 0.68) -> np.ndarray:
    x0, y0, x1, y1 = limites_roi_central(imagen_bgr, escala)
    return imagen_bgr[y0:y1, x0:x1]


def dibujar_roi_central(imagen_bgr: np.ndarray, escala: float = 0.68) -> np.ndarray:
    salida = imagen_bgr.copy()
    x0, y0, x1, y1 = limites_roi_central(salida, escala)
    cv2.rectangle(salida, (x0, y0), (x1, y1), (64, 57, 52), 2)
    cv2.putText(
        salida,
        "Coloca la mano dentro del recuadro",
        (x0, max(24, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (64, 57, 52),
        2,
        cv2.LINE_AA,
    )
    return salida


def crear_mascara_piel(imagen_bgr: np.ndarray, rango_hsv: RangoHSV = RangoHSV()) -> np.ndarray:
    """Convierte a HSV y deja en blanco los pixeles compatibles con piel."""
    hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)
    inferior = np.array(rango_hsv.inferior, dtype=np.uint8)
    superior = np.array(rango_hsv.superior, dtype=np.uint8)
    mascara = cv2.inRange(hsv, inferior, superior)

    # Limpieza morfologica: quita puntos pequenos y rellena huecos de la mano.
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo, iterations=1)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, nucleo, iterations=2)
    mascara = cv2.GaussianBlur(mascara, (5, 5), 0)
    _, mascara = cv2.threshold(mascara, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mascara


def extraer_roi_mano(
    imagen_bgr: np.ndarray,
    mascara: np.ndarray,
    area_minima: int = 1200,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Busca el contorno mas grande y devuelve el recorte de mano y mascara."""
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    contorno = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(contorno) < area_minima:
        return None

    x, y, ancho, alto = cv2.boundingRect(contorno)
    margen = int(max(ancho, alto) * 0.12)
    y0 = max(0, y - margen)
    y1 = min(imagen_bgr.shape[0], y + alto + margen)
    x0 = max(0, x - margen)
    x1 = min(imagen_bgr.shape[1], x + ancho + margen)
    return imagen_bgr[y0:y1, x0:x1], mascara[y0:y1, x0:x1]


def detectar_bordes(mascara: np.ndarray, modo: ModoBordes = "canny") -> np.ndarray:
    """Obtiene la forma interna/externa de la mano desde la mascara."""
    if modo == "sobel":
        sx = cv2.Sobel(mascara, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(mascara, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.convertScaleAbs(cv2.magnitude(sx, sy))

    suavizada = cv2.GaussianBlur(mascara, (5, 5), 0)
    return cv2.Canny(suavizada, 60, 160)


def redimensionar_cuadrado(imagen: np.ndarray, tamano: int = 64) -> np.ndarray:
    """Centra cualquier recorte en un lienzo cuadrado antes de redimensionar."""
    alto, ancho = imagen.shape[:2]
    lado = max(alto, ancho)
    if imagen.ndim == 2:
        lienzo = np.zeros((lado, lado), dtype=imagen.dtype)
    else:
        lienzo = np.zeros((lado, lado, imagen.shape[2]), dtype=imagen.dtype)
    y = (lado - alto) // 2
    x = (lado - ancho) // 2
    lienzo[y : y + alto, x : x + ancho] = imagen
    return cv2.resize(lienzo, (tamano, tamano), interpolation=cv2.INTER_AREA)


def preprocesar_imagen(
    imagen_bgr: np.ndarray,
    config: ConfiguracionPreprocesamiento = ConfiguracionPreprocesamiento(),
) -> tuple[np.ndarray, np.ndarray] | None:
    """Devuelve (bordes_64x64, mascara_64x64) o None si no hay mano valida."""
    mascara = crear_mascara_piel(imagen_bgr, config.rango_hsv)
    roi = extraer_roi_mano(imagen_bgr, mascara, config.area_minima)
    if roi is None:
        return None

    _, mascara_roi = roi
    bordes = detectar_bordes(mascara_roi, config.modo_bordes)
    return (
        redimensionar_cuadrado(bordes, config.tamano_imagen),
        redimensionar_cuadrado(mascara_roi, config.tamano_imagen),
    )


def dibujar_caja_mano(
    imagen_bgr: np.ndarray,
    config: ConfiguracionPreprocesamiento = ConfiguracionPreprocesamiento(),
) -> np.ndarray:
    mascara = crear_mascara_piel(imagen_bgr, config.rango_hsv)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return imagen_bgr

    contorno = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(contorno) < config.area_minima:
        return imagen_bgr

    x, y, ancho, alto = cv2.boundingRect(contorno)
    salida = imagen_bgr.copy()
    cv2.rectangle(salida, (x, y), (x + ancho, y + alto), (87, 80, 73), 2)
    return salida


# Alias para mantener compatibilidad con nombres anteriores.
HSVRange = RangoHSV
PreprocessConfig = ConfiguracionPreprocesamiento
center_roi_bounds = limites_roi_central
center_roi = recortar_roi_central
draw_center_roi = dibujar_roi_central
skin_mask_bgr = crear_mascara_piel
largest_hand_roi = extraer_roi_mano
hand_edges = detectar_bordes
square_resize = redimensionar_cuadrado
preprocess_bgr = preprocesar_imagen
draw_hand_box = dibujar_caja_mano
