import unittest
from pathlib import Path

import numpy as np
import torch

from sign_recognition.cnn_model import (
    RedConvolucionalSenas,
    cargar_metadatos_modelo,
    cargar_modelo_cnn,
    crear_modelo_hibrido,
    guardar_modelo_cnn,
)
from sign_recognition.preprocess import ConfiguracionPreprocesamiento
from sign_recognition.shape_features import NUMERO_CARACTERISTICAS_FORMA


class ModeloTests(unittest.TestCase):
    def test_forma_de_salida(self) -> None:
        modelo = RedConvolucionalSenas(numero_clases=8)
        salida = modelo(torch.zeros((4, 2, 64, 64)))
        self.assertEqual(tuple(salida.shape), (4, 8))

    def test_forma_de_salida_hibrida(self) -> None:
        modelo = crear_modelo_hibrido(
            numero_clases=8,
            media_forma=np.zeros(NUMERO_CARACTERISTICAS_FORMA, dtype=np.float32),
            desviacion_forma=np.ones(NUMERO_CARACTERISTICAS_FORMA, dtype=np.float32),
        )
        salida = modelo(
            torch.zeros((4, 2, 64, 64)),
            torch.zeros((4, NUMERO_CARACTERISTICAS_FORMA)),
        )
        self.assertEqual(tuple(salida.shape), (4, 8))

    def test_checkpoint_conserva_configuracion_y_metadatos(self) -> None:
        modelo = RedConvolucionalSenas(numero_clases=2)
        config = ConfiguracionPreprocesamiento()
        ruta = Path(__file__).parent / "checkpoint_prueba.pt"

        try:
            guardar_modelo_cnn(
                ruta,
                modelo,
                ["A", "NONE"],
                config,
                0.9,
                {"umbral_confianza": 0.72},
            )
            cargado, etiquetas, config_cargada = cargar_modelo_cnn(ruta)
            metadatos = cargar_metadatos_modelo(ruta)
        finally:
            ruta.unlink(missing_ok=True)

        self.assertIsInstance(cargado, RedConvolucionalSenas)
        self.assertEqual(etiquetas, ["A", "NONE"])
        self.assertEqual(config_cargada, config)
        self.assertEqual(metadatos["umbral_confianza"], 0.72)

    def test_checkpoint_hibrido_conserva_normalizacion(self) -> None:
        media = np.arange(NUMERO_CARACTERISTICAS_FORMA, dtype=np.float32)
        desviacion = np.full(NUMERO_CARACTERISTICAS_FORMA, 2.0, dtype=np.float32)
        modelo = crear_modelo_hibrido(2, media, desviacion)
        ruta = Path(__file__).parent / "checkpoint_hibrido_prueba.pt"

        try:
            guardar_modelo_cnn(
                ruta,
                modelo,
                ["G", "H"],
                ConfiguracionPreprocesamiento(),
                0.9,
            )
            cargado, _, _ = cargar_modelo_cnn(ruta)
        finally:
            ruta.unlink(missing_ok=True)

        self.assertEqual(cargado.numero_caracteristicas_forma, NUMERO_CARACTERISTICAS_FORMA)
        np.testing.assert_allclose(cargado.media_forma.numpy(), media)
        np.testing.assert_allclose(cargado.desviacion_forma.numpy(), desviacion)


if __name__ == "__main__":
    unittest.main()
