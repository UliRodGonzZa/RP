from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iterar_imagenes(directorio_datos: Path):
    """Recorre `data/raw` y entrega pares (etiqueta, ruta_imagen)."""
    for carpeta_clase in sorted(p for p in directorio_datos.iterdir() if p.is_dir()):
        for ruta_imagen in sorted(carpeta_clase.rglob("*")):
            if ruta_imagen.suffix.lower() in EXTENSIONES_IMAGEN:
                yield carpeta_clase.name, ruta_imagen


def identificar_grupo_captura(
    directorio_datos: Path,
    etiqueta: str,
    ruta_imagen: Path,
    tamano_bloque: int = 10,
) -> str:
    """Identifica una sesion real o un bloque consecutivo de capturas antiguas."""
    relativa = ruta_imagen.relative_to(directorio_datos / etiqueta)
    if len(relativa.parts) > 1:
        return f"{etiqueta}/{relativa.parts[0]}"

    coincidencia = re.search(r"(\d+)$", ruta_imagen.stem)
    if coincidencia:
        indice = max(int(coincidencia.group(1)) - 1, 0)
        return f"{etiqueta}/bloque_{indice // tamano_bloque:04d}"
    return f"{etiqueta}/{ruta_imagen.stem}"


def dividir_indices_por_grupo(
    etiquetas: np.ndarray,
    grupos: np.ndarray,
    proporcion_validacion: float = 0.15,
    proporcion_prueba: float = 0.15,
    semilla: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide cada clase sin permitir que un grupo aparezca en dos conjuntos."""
    if proporcion_validacion <= 0 or proporcion_prueba <= 0:
        raise ValueError("Las proporciones de validacion y prueba deben ser positivas.")
    if proporcion_validacion + proporcion_prueba >= 1:
        raise ValueError("La suma de validacion y prueba debe ser menor que 1.")

    rng = random.Random(semilla)
    conjuntos: dict[str, list[int]] = {"entrenamiento": [], "validacion": [], "prueba": []}

    for etiqueta in sorted(set(etiquetas.tolist())):
        indices_clase = np.flatnonzero(etiquetas == etiqueta)
        grupos_clase = sorted(set(grupos[indices_clase].tolist()))
        if len(grupos_clase) < 3:
            raise ValueError(
                f"La clase {etiqueta!r} necesita al menos 3 sesiones o bloques; "
                f"solo tiene {len(grupos_clase)}."
            )

        rng.shuffle(grupos_clase)
        cantidad = len(grupos_clase)
        cantidad_prueba = max(1, round(cantidad * proporcion_prueba))
        cantidad_validacion = max(1, round(cantidad * proporcion_validacion))
        if cantidad_prueba + cantidad_validacion >= cantidad:
            cantidad_validacion = 1
            cantidad_prueba = 1

        grupos_prueba = set(grupos_clase[:cantidad_prueba])
        grupos_validacion = set(grupos_clase[cantidad_prueba : cantidad_prueba + cantidad_validacion])

        for indice in indices_clase:
            grupo = grupos[indice]
            if grupo in grupos_prueba:
                conjuntos["prueba"].append(int(indice))
            elif grupo in grupos_validacion:
                conjuntos["validacion"].append(int(indice))
            else:
                conjuntos["entrenamiento"].append(int(indice))

    return tuple(
        np.array(conjuntos[nombre], dtype=np.int64)
        for nombre in ("entrenamiento", "validacion", "prueba")
    )


# Alias temporal por compatibilidad con scripts viejos que aun pudieran importarlo.
iter_images = iterar_imagenes
