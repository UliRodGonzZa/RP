from __future__ import annotations

import sys
from dataclasses import dataclass

import cv2


BACKENDS_CAMARA = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}


@dataclass(frozen=True)
class CamaraAbierta:
    captura: cv2.VideoCapture
    nombre_backend: str


def candidatos_backend(preferido: str = "auto") -> list[tuple[str, int]]:
    """Devuelve backends de OpenCV en el orden mas estable para el sistema."""
    if preferido != "auto":
        if preferido not in BACKENDS_CAMARA:
            raise ValueError(f"Backend de camara no soportado: {preferido}")
        return [(preferido, BACKENDS_CAMARA[preferido])]

    # En Windows, DirectShow suele ser mas estable que MSMF para webcams.
    if sys.platform.startswith("win"):
        return [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("any", cv2.CAP_ANY)]
    return [("any", cv2.CAP_ANY)]


def abrir_camara(indice_camara: int = 0, backend_preferido: str = "auto") -> CamaraAbierta:
    """Abre la camara y comprueba que realmente pueda entregar frames."""
    for nombre_backend, backend in candidatos_backend(backend_preferido):
        captura = cv2.VideoCapture(indice_camara, backend)
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        captura.set(cv2.CAP_PROP_FPS, 30)

        lectura_correcta, _ = captura.read()
        if captura.isOpened() and lectura_correcta:
            return CamaraAbierta(captura=captura, nombre_backend=nombre_backend)

        captura.release()

    raise RuntimeError("No se pudo abrir la camara ni leer frames.")


# Alias temporal por compatibilidad.
open_camera = abrir_camara
