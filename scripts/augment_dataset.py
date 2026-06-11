"""
Genera un numero fijo de imagenes aumentadas por clase variando
rotacion e iluminacion, y las guarda en una subcarpeta nueva.

Caso A — capturas SIN --session (imagenes directo en data/raw/A/):
    python scripts/augment_dataset.py --count 500

Caso B — capturas CON --session rodrigo_s1:
    python scripts/augment_dataset.py --session rodrigo_s1 --count 500
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def rotar(imagen: np.ndarray, angulo: float) -> np.ndarray:
    alto, ancho = imagen.shape[:2]
    M = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo, 1.0)
    return cv2.warpAffine(imagen, M, (ancho, alto), borderMode=cv2.BORDER_REFLECT)


def ajustar_brillo(imagen: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augmentar(imagen: np.ndarray) -> np.ndarray:
    return ajustar_brillo(rotar(imagen, random.uniform(-15.0, 15.0)), random.uniform(0.75, 1.30))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera imagenes aumentadas a partir de un dataset capturado."
    )
    parser.add_argument("--session", default=None, help="Sesion origen (opcional). Si no se indica, usa todas las imagenes directas de cada clase.")
    parser.add_argument("--out-session", default="aug", help="Nombre de la subcarpeta de salida (default: aug)")
    parser.add_argument("--count", type=int, default=500, help="Total de imagenes a generar por clase (default: 500)")
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    total_generadas = 0

    print(f"\nGenerando {args.count} imagenes aumentadas por clase → subcarpeta '{args.out_session}'\n")

    for carpeta_clase in sorted(args.data.iterdir()):
        if not carpeta_clase.is_dir():
            continue

        # Determina de donde tomar las imagenes origen
        if args.session:
            carpeta_origen = carpeta_clase / args.session
            if not carpeta_origen.exists():
                print(f"  [SKIP] {carpeta_clase.name:6s} — no se encontro sesion '{args.session}'")
                continue
            imagenes = sorted(p for p in carpeta_origen.iterdir() if p.suffix.lower() in EXTENSIONES)
        else:
            # Sin sesion: toma solo las imagenes directas (no entra a subcarpetas)
            imagenes = sorted(p for p in carpeta_clase.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONES)

        if not imagenes:
            print(f"  [SKIP] {carpeta_clase.name:6s} — sin imagenes")
            continue

        carpeta_salida = carpeta_clase / args.out_session
        carpeta_salida.mkdir(exist_ok=True)

        seleccionadas = random.choices(imagenes, k=args.count)

        for indice, ruta in enumerate(seleccionadas, start=1):
            imagen = cv2.imread(str(ruta))
            if imagen is None:
                continue
            cv2.imwrite(str(carpeta_salida / f"{carpeta_clase.name}_{indice:04d}.jpg"), augmentar(imagen))

        total_generadas += args.count
        print(f"  [{carpeta_clase.name:6s}] {len(imagenes):4d} imagenes origen → {args.count} aumentadas generadas")

    print(f"\n{'─' * 50}")
    print(f"  Total generadas : {total_generadas}")
    print(f"  Subcarpeta      : */{args.out_session}/")
    print(f"{'─' * 50}\n")
    print("Listo. Ahora entrena con:")
    print("  .venv\\Scripts\\python.exe scripts\\train_cnn.py --data data\\raw --model-out models\\hybrid\\sign_hybrid.pt --epochs 50 --model-type hybrid")


if __name__ == "__main__":
    main()

