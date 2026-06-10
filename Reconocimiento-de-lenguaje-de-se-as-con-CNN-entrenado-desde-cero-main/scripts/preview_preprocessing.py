from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sign_recognition.dataset import iterar_imagenes
from sign_recognition.preprocess import (
    ConfiguracionPreprocesamiento,
    crear_mascara_piel,
    detectar_bordes,
    extraer_roi_mano,
    redimensionar_cuadrado,
)


def crear_panel(titulo: str, imagen_bgr: np.ndarray, ancho: int = 280, alto: int = 280) -> np.ndarray:
    if imagen_bgr.ndim == 2:
        imagen_bgr = cv2.cvtColor(imagen_bgr, cv2.COLOR_GRAY2BGR)

    panel = np.full((alto + 36, ancho, 3), 248, dtype=np.uint8)
    panel[36:, :] = cv2.resize(imagen_bgr, (ancho, alto), interpolation=cv2.INTER_AREA)
    cv2.putText(panel, titulo, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (33, 37, 41), 2, cv2.LINE_AA)
    return panel


def previsualizar_imagen(imagen_bgr: np.ndarray, config: ConfiguracionPreprocesamiento) -> tuple[np.ndarray, bool]:
    mascara = crear_mascara_piel(imagen_bgr, config.rango_hsv)
    roi = extraer_roi_mano(imagen_bgr, mascara, config.area_minima)

    panel_original = crear_panel("Original", imagen_bgr)
    panel_mascara = crear_panel("Mascara HSV", mascara)

    if roi is None:
        vacio = np.full((316, 280, 3), 248, dtype=np.uint8)
        cv2.putText(vacio, "Omitida", (82, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (73, 80, 87), 2, cv2.LINE_AA)
        cv2.putText(vacio, "sin mano/area baja", (32, 188), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (108, 117, 125), 2, cv2.LINE_AA)
        return np.hstack([panel_original, panel_mascara, vacio, vacio.copy()]), False

    imagen_roi, mascara_roi = roi
    bordes = detectar_bordes(mascara_roi, config.modo_bordes)
    panel_roi = crear_panel("ROI detectado", imagen_roi)
    panel_bordes = crear_panel("Bordes", bordes)

    mascara_64 = redimensionar_cuadrado(mascara_roi, config.tamano_imagen)
    bordes_64 = redimensionar_cuadrado(bordes, config.tamano_imagen)
    entrada = np.hstack(
        [
            cv2.resize(mascara_64, (140, 140), interpolation=cv2.INTER_NEAREST),
            cv2.resize(bordes_64, (140, 140), interpolation=cv2.INTER_NEAREST),
        ]
    )
    panel_entrada = crear_panel("Entrada CNN", entrada)
    return np.hstack([panel_original, panel_mascara, panel_roi, panel_bordes, panel_entrada]), True


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta previsualizaciones del preprocesamiento por clase.")
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("models/previews"))
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--edge-mode", choices=["canny", "sobel"], default="canny")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = ConfiguracionPreprocesamiento(modo_bordes=args.edge_mode)
    conteos: dict[str, int] = {}
    aceptadas: dict[str, int] = {}

    for etiqueta, ruta_imagen in iterar_imagenes(args.data):
        conteos.setdefault(etiqueta, 0)
        aceptadas.setdefault(etiqueta, 0)
        if conteos[etiqueta] >= args.per_class:
            continue

        imagen = cv2.imread(str(ruta_imagen))
        if imagen is None:
            continue

        panel, aceptada = previsualizar_imagen(imagen, config)
        estado = "usada" if aceptada else "omitida"
        carpeta_salida = args.out / etiqueta
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        ruta_salida = carpeta_salida / f"{conteos[etiqueta] + 1:03d}_{estado}_{ruta_imagen.stem}.jpg"
        cv2.imwrite(str(ruta_salida), panel)
        conteos[etiqueta] += 1
        aceptadas[etiqueta] += int(aceptada)

    print(f"Previews guardados en: {args.out}")
    print("Clase | Previews | Aceptadas | Omitidas")
    for etiqueta in sorted(conteos):
        omitidas = conteos[etiqueta] - aceptadas[etiqueta]
        print(f"{etiqueta:>5} | {conteos[etiqueta]:>8} | {aceptadas[etiqueta]:>9} | {omitidas:>8}")


if __name__ == "__main__":
    main()
