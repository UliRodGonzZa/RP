from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


NOMBRES_CARACTERISTICAS_FORMA = [
    "hu_1",
    "hu_2",
    "hu_3",
    "hu_4",
    "hu_5",
    "hu_6",
    "hu_7",
    "relacion_aspecto",
    "extension",
    "solidez",
    "circularidad",
]
NUMERO_CARACTERISTICAS_FORMA = len(NOMBRES_CARACTERISTICAS_FORMA)


@dataclass(frozen=True)
class AnalisisGeometrico:
    contorno: np.ndarray
    envolvente: np.ndarray
    caja_rotada: np.ndarray
    centroide: tuple[int, int]
    eje_mayor: tuple[tuple[int, int], tuple[int, int]]
    eje_menor: tuple[tuple[int, int], tuple[int, int]]
    relacion_aspecto: float
    extension: float
    solidez: float
    circularidad: float


def analizar_geometria(mascara: np.ndarray) -> AnalisisGeometrico | None:
    """Obtiene las construcciones geometricas principales de una silueta."""
    if mascara.ndim != 2:
        raise ValueError("La mascara debe tener dos dimensiones.")

    mascara_binaria = (mascara > 0).astype(np.uint8) * 255
    contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None

    contorno = max(contornos, key=cv2.contourArea)
    area = float(cv2.contourArea(contorno))
    if area <= 0:
        return None

    perimetro = float(cv2.arcLength(contorno, True))
    _, _, ancho, alto = cv2.boundingRect(contorno)
    envolvente = cv2.convexHull(contorno)
    area_envolvente = float(cv2.contourArea(envolvente))
    momentos = cv2.moments(contorno)
    centroide = (
        int(momentos["m10"] / momentos["m00"]),
        int(momentos["m01"] / momentos["m00"]),
    )

    rectangulo_rotado = cv2.minAreaRect(contorno)
    centro_rotado, dimensiones, angulo = rectangulo_rotado
    ancho_rotado, alto_rotado = dimensiones
    if ancho_rotado < alto_rotado:
        angulo += 90.0
        longitud_mayor, longitud_menor = alto_rotado, ancho_rotado
    else:
        longitud_mayor, longitud_menor = ancho_rotado, alto_rotado

    radianes = np.deg2rad(angulo)
    vector_mayor = np.array([np.cos(radianes), np.sin(radianes)])
    vector_menor = np.array([-np.sin(radianes), np.cos(radianes)])
    centro = np.array(centro_rotado)

    def extremos(vector: np.ndarray, longitud: float) -> tuple[tuple[int, int], tuple[int, int]]:
        mitad = vector * longitud * 0.5
        return tuple(np.rint(centro - mitad).astype(int)), tuple(np.rint(centro + mitad).astype(int))

    return AnalisisGeometrico(
        contorno=contorno,
        envolvente=envolvente,
        caja_rotada=np.rint(cv2.boxPoints(rectangulo_rotado)).astype(np.int32),
        centroide=centroide,
        eje_mayor=extremos(vector_mayor, longitud_mayor),
        eje_menor=extremos(vector_menor, longitud_menor),
        relacion_aspecto=ancho / max(alto, 1),
        extension=area / max(ancho * alto, 1),
        solidez=area / max(area_envolvente, 1.0),
        circularidad=4.0 * np.pi * area / max(perimetro * perimetro, 1.0),
    )


def dibujar_analisis_geometrico(
    imagen_bgr: np.ndarray,
    mascara: np.ndarray,
) -> tuple[np.ndarray, AnalisisGeometrico | None]:
    """Dibuja contorno, envolvente, caja, centroide y ejes de la silueta."""
    salida = imagen_bgr.copy()
    analisis = analizar_geometria(mascara)
    if analisis is None:
        return salida, None

    cv2.drawContours(salida, [analisis.envolvente], -1, (80, 220, 120), 2)
    cv2.drawContours(salida, [analisis.contorno], -1, (255, 210, 70), 2)
    cv2.polylines(salida, [analisis.caja_rotada], True, (40, 145, 255), 2)
    cv2.line(salida, *analisis.eje_mayor, (220, 80, 220), 2, cv2.LINE_AA)
    cv2.line(salida, *analisis.eje_menor, (90, 220, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(
        salida,
        analisis.centroide,
        (60, 60, 255),
        cv2.MARKER_CROSS,
        16,
        2,
        cv2.LINE_AA,
    )
    return salida, analisis


def extraer_caracteristicas_forma(mascara: np.ndarray) -> np.ndarray:
    """Describe globalmente una silueta binaria con momentos y geometria."""
    if mascara.ndim != 2:
        raise ValueError("La mascara debe tener dos dimensiones.")

    mascara_binaria = (mascara > 0).astype(np.uint8) * 255
    analisis = analizar_geometria(mascara_binaria)
    if analisis is None:
        return np.zeros(NUMERO_CARACTERISTICAS_FORMA, dtype=np.float32)

    momentos = cv2.moments(mascara_binaria, binaryImage=True)
    hu = cv2.HuMoments(momentos).flatten()
    hu_log = -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-12))

    caracteristicas = np.concatenate(
        [
            hu_log,
            np.array(
                [
                    analisis.relacion_aspecto,
                    analisis.extension,
                    analisis.solidez,
                    analisis.circularidad,
                ]
            ),
        ]
    )
    return np.nan_to_num(caracteristicas, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
