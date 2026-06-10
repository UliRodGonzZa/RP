from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from sign_recognition.preprocess import ConfiguracionPreprocesamiento, RangoHSV, preprocesar_imagen
from sign_recognition.shape_features import NUMERO_CARACTERISTICAS_FORMA


class RedConvolucionalSenas(nn.Module):
    """CNN pequena con una rama opcional de descriptores de forma.

    Entrada:
        canal 0: mascara HSV de la mano
        canal 1: bordes de la mano
    """

    def __init__(
        self,
        numero_clases: int,
        numero_caracteristicas_forma: int = 0,
        media_forma: np.ndarray | None = None,
        desviacion_forma: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.numero_caracteristicas_forma = numero_caracteristicas_forma
        self.extractor = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        dimension_clasificador = 64
        if numero_caracteristicas_forma:
            self.proyector_forma = nn.Sequential(
                nn.Linear(numero_caracteristicas_forma, 32),
                nn.LayerNorm(32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.15),
            )
            dimension_clasificador += 32
            media = np.zeros(numero_caracteristicas_forma, dtype=np.float32) if media_forma is None else media_forma
            desviacion = (
                np.ones(numero_caracteristicas_forma, dtype=np.float32)
                if desviacion_forma is None
                else desviacion_forma
            )
            self.register_buffer("media_forma", torch.as_tensor(media, dtype=torch.float32))
            self.register_buffer("desviacion_forma", torch.as_tensor(desviacion, dtype=torch.float32))

        self.clasificador = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(dimension_clasificador, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, numero_clases),
        )

    def forward(
        self,
        x: torch.Tensor,
        caracteristicas_forma: torch.Tensor | None = None,
    ) -> torch.Tensor:
        representacion = torch.flatten(self.extractor(x), 1)
        if self.numero_caracteristicas_forma:
            if caracteristicas_forma is None:
                raise ValueError("El modelo hibrido requiere caracteristicas de forma.")
            forma_normalizada = (caracteristicas_forma - self.media_forma) / self.desviacion_forma
            representacion_forma = self.proyector_forma(forma_normalizada)
            representacion = torch.cat([representacion, representacion_forma], dim=1)
        return self.clasificador(representacion)


def configuracion_a_diccionario(config: ConfiguracionPreprocesamiento) -> dict:
    return {
        "tamano_imagen": config.tamano_imagen,
        "hsv_inferior": config.rango_hsv.inferior,
        "hsv_superior": config.rango_hsv.superior,
        "modo_bordes": config.modo_bordes,
        "area_minima": config.area_minima,
    }


def configuracion_desde_diccionario(datos: dict) -> ConfiguracionPreprocesamiento:
    return ConfiguracionPreprocesamiento(
        tamano_imagen=int(datos.get("tamano_imagen", datos.get("image_size", 64))),
        rango_hsv=RangoHSV(
            inferior=tuple(datos.get("hsv_inferior", datos.get("hsv_lower", (0, 25, 45)))),
            superior=tuple(datos.get("hsv_superior", datos.get("hsv_upper", (25, 255, 255)))),
        ),
        modo_bordes=datos.get("modo_bordes", datos.get("edge_mode", "canny")),
        area_minima=int(datos.get("area_minima", datos.get("min_area", 1200))),
    )


def entrada_cnn_desde_bgr(imagen_bgr: np.ndarray, config: ConfiguracionPreprocesamiento) -> np.ndarray | None:
    """Convierte una imagen BGR en tensor numpy de 2 canales para la CNN."""
    procesada = preprocesar_imagen(imagen_bgr, config)
    if procesada is None:
        return None

    bordes, mascara = procesada
    return np.stack([mascara, bordes], axis=0).astype("float32") / 255.0


def aumentar_entrada_cnn(muestra: np.ndarray) -> np.ndarray:
    """Genera variaciones artificiales para mejorar con pocos datos."""
    canales, alto, ancho = muestra.shape
    angulo = float(np.random.uniform(-14, 14))
    escala = float(np.random.uniform(0.88, 1.12))
    tx = float(np.random.uniform(-0.08, 0.08) * ancho)
    ty = float(np.random.uniform(-0.08, 0.08) * alto)

    matriz = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo, escala)
    matriz[:, 2] += [tx, ty]

    aumentada = np.empty_like(muestra)
    for indice in range(canales):
        aumentada[indice] = cv2.warpAffine(
            muestra[indice],
            matriz,
            (ancho, alto),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # Simula pequenas variaciones de segmentacion HSV.
    if np.random.random() < 0.35:
        tamano_nucleo = int(np.random.choice([2, 3]))
        nucleo = np.ones((tamano_nucleo, tamano_nucleo), dtype=np.uint8)
        canal_mascara = np.clip(aumentada[0] * 255, 0, 255).astype("uint8")
        if np.random.random() < 0.5:
            canal_mascara = cv2.erode(canal_mascara, nucleo, iterations=1)
        else:
            canal_mascara = cv2.dilate(canal_mascara, nucleo, iterations=1)
        aumentada[0] = canal_mascara.astype("float32") / 255.0

    if np.random.random() < 0.30:
        ruido = np.random.normal(0, 0.025, aumentada.shape).astype("float32")
        aumentada = aumentada + ruido

    return np.clip(aumentada, 0.0, 1.0).astype("float32")


def guardar_modelo_cnn(
    ruta: Path,
    modelo: RedConvolucionalSenas,
    etiquetas: list[str],
    config: ConfiguracionPreprocesamiento,
    exactitud: float,
    metadatos: dict | None = None,
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "version_formato": 3,
            "arquitectura": {
                "numero_caracteristicas_forma": modelo.numero_caracteristicas_forma,
            },
            "estado_modelo": modelo.state_dict(),
            "etiquetas": etiquetas,
            "configuracion": configuracion_a_diccionario(config),
            "exactitud": exactitud,
            "metadatos": metadatos or {},
        },
        ruta,
    )


def cargar_modelo_cnn(
    ruta: Path,
    dispositivo: torch.device | str = "cpu",
) -> tuple[RedConvolucionalSenas, list[str], ConfiguracionPreprocesamiento]:
    punto_control = cargar_punto_control(ruta, dispositivo)
    etiquetas = list(punto_control.get("etiquetas", punto_control.get("labels")))
    arquitectura = punto_control.get("arquitectura", {})
    numero_caracteristicas_forma = int(arquitectura.get("numero_caracteristicas_forma", 0))
    modelo = RedConvolucionalSenas(
        numero_clases=len(etiquetas),
        numero_caracteristicas_forma=numero_caracteristicas_forma,
    )
    estado = punto_control.get("estado_modelo", punto_control.get("model_state"))

    # Permite abrir modelos guardados antes de traducir los nombres internos.
    if any(clave.startswith("features.") or clave.startswith("classifier.") for clave in estado):
        estado = {
            clave.replace("features.", "extractor.").replace("classifier.", "clasificador."): valor
            for clave, valor in estado.items()
        }

    modelo.load_state_dict(estado)
    modelo.to(dispositivo)
    modelo.eval()
    return modelo, etiquetas, configuracion_desde_diccionario(
        punto_control.get("configuracion", punto_control.get("config"))
    )


def crear_modelo_hibrido(
    numero_clases: int,
    media_forma: np.ndarray,
    desviacion_forma: np.ndarray,
) -> RedConvolucionalSenas:
    return RedConvolucionalSenas(
        numero_clases=numero_clases,
        numero_caracteristicas_forma=NUMERO_CARACTERISTICAS_FORMA,
        media_forma=media_forma,
        desviacion_forma=desviacion_forma,
    )


def cargar_punto_control(ruta: Path, dispositivo: torch.device | str = "cpu") -> dict:
    """Carga un checkpoint usando el modo seguro cuando PyTorch lo soporta."""
    try:
        return torch.load(ruta, map_location=dispositivo, weights_only=True)
    except TypeError:
        return torch.load(ruta, map_location=dispositivo)


def cargar_metadatos_modelo(ruta: Path) -> dict:
    punto_control = cargar_punto_control(ruta, "cpu")
    return dict(punto_control.get("metadatos", {}))


# Alias para compatibilidad con nombres anteriores.
SmallSignCNN = RedConvolucionalSenas
cnn_input_from_bgr = entrada_cnn_desde_bgr
augment_cnn_input = aumentar_entrada_cnn
save_cnn_checkpoint = guardar_modelo_cnn
load_cnn_checkpoint = cargar_modelo_cnn
