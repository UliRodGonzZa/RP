from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sign_recognition.camera import abrir_camara
from sign_recognition.preprocess import (
    ConfiguracionPreprocesamiento,
    crear_mascara_piel,
    dibujar_caja_mano,
    dibujar_roi_central,
    limites_roi_central,
    preprocesar_imagen,
    recortar_roi_central,
)


def crear_vista_captura(
    frame,
    roi_mano,
    config: ConfiguracionPreprocesamiento,
    etiqueta: str,
    capturadas: int,
    total: int,
):
    """Une video, mascara HSV y entrada procesada en una sola ventana."""
    vista = dibujar_roi_central(frame)
    x0, y0, x1, y1 = limites_roi_central(vista)
    vista_roi = dibujar_caja_mano(vista[y0:y1, x0:x1].copy(), config)
    vista[y0:y1, x0:x1] = vista_roi

    mascara = crear_mascara_piel(roi_mano, config.rango_hsv)
    procesada = preprocesar_imagen(roi_mano, config)
    mascara_bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)

    if procesada is None:
        entrada_bgr = np.full_like(mascara_bgr, 248)
        cv2.putText(entrada_bgr, "Sin mano valida", (20, entrada_bgr.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (73, 80, 87), 2, cv2.LINE_AA)
    else:
        bordes, _ = procesada
        entrada_bgr = cv2.cvtColor(bordes, cv2.COLOR_GRAY2BGR)

    lado = vista.shape[0] // 2
    panel_mascara = cv2.resize(mascara_bgr, (lado, lado), interpolation=cv2.INTER_NEAREST)
    panel_entrada = cv2.resize(entrada_bgr, (lado, lado), interpolation=cv2.INTER_NEAREST)
    panel_derecho = np.vstack([panel_mascara, panel_entrada])
    panel_derecho = cv2.resize(panel_derecho, (lado, vista.shape[0]))

    cv2.putText(panel_derecho, "Mascara HSV", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (248, 249, 250), 2, cv2.LINE_AA)
    cv2.putText(panel_derecho, "Entrada CNN", (12, lado + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (248, 249, 250), 2, cv2.LINE_AA)

    combinada = np.hstack([vista, panel_derecho])
    cv2.putText(
        combinada,
        f"Clase: {etiqueta} | Capturadas: {capturadas}/{total} | c=capturar q=salir",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return combinada, procesada is not None


def siguiente_indice(carpeta_etiqueta: Path, etiqueta: str) -> int:
    existentes = sorted(carpeta_etiqueta.glob(f"{etiqueta}_*.jpg"))
    if not existentes:
        return 1
    return max(int(ruta.stem.split("_")[-1]) for ruta in existentes) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Captura muestras para entrenar una sena.")
    parser.add_argument("--label", required=True, help="Etiqueta de la sena, por ejemplo A.")
    parser.add_argument("--count", type=int, default=80, help="Numero de fotos a capturar.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de camara.")
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    parser.add_argument("--out", type=Path, default=Path("data/raw"), help="Directorio de salida.")
    parser.add_argument(
        "--session",
        default=None,
        help="Identificador de persona/sesion, por ejemplo persona01_sesion01.",
    )
    args = parser.parse_args()

    etiqueta = args.label.strip().upper()
    carpeta_etiqueta = args.out / etiqueta
    if args.session:
        sesion = args.session.strip().replace(" ", "_")
        if not sesion or any(caracter in sesion for caracter in '<>:"/\\|?*'):
            raise ValueError("El identificador de sesion contiene caracteres no validos.")
        carpeta_etiqueta /= sesion
    carpeta_etiqueta.mkdir(parents=True, exist_ok=True)

    camara = abrir_camara(args.camera, args.backend)
    captura = camara.captura
    print(f"Camara abierta con backend: {camara.nombre_backend}")

    config = ConfiguracionPreprocesamiento()
    indice = siguiente_indice(carpeta_etiqueta, etiqueta)
    capturadas = 0

    try:
        while capturadas < args.count:
            lectura_correcta, frame = captura.read()
            if not lectura_correcta:
                break

            roi_mano = recortar_roi_central(frame)
            vista, mano_valida = crear_vista_captura(frame, roi_mano, config, etiqueta, capturadas, args.count)
            cv2.imshow("Captura de muestras", vista)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("c"):
                if not mano_valida:
                    print("No se detecto una mano clara; ajusta iluminacion, fondo o HSV.")
                    continue

                ruta_salida = carpeta_etiqueta / f"{etiqueta}_{indice:04d}.jpg"
                cv2.imwrite(str(ruta_salida), roi_mano)
                indice += 1
                capturadas += 1
                print(f"Guardada: {ruta_salida}")
    finally:
        captura.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
