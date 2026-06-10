import unittest

import cv2
import numpy as np

from sign_recognition.shape_features import (
    NUMERO_CARACTERISTICAS_FORMA,
    analizar_geometria,
    dibujar_analisis_geometrico,
    extraer_caracteristicas_forma,
)


class CaracteristicasFormaTests(unittest.TestCase):
    def test_devuelve_vector_finito(self) -> None:
        mascara = np.zeros((64, 64), dtype=np.uint8)
        cv2.rectangle(mascara, (18, 10), (45, 54), 255, thickness=-1)

        caracteristicas = extraer_caracteristicas_forma(mascara)

        self.assertEqual(caracteristicas.shape, (NUMERO_CARACTERISTICAS_FORMA,))
        self.assertTrue(np.isfinite(caracteristicas).all())

    def test_mascara_vacia_devuelve_ceros(self) -> None:
        caracteristicas = extraer_caracteristicas_forma(np.zeros((64, 64), dtype=np.uint8))
        np.testing.assert_array_equal(
            caracteristicas,
            np.zeros(NUMERO_CARACTERISTICAS_FORMA, dtype=np.float32),
        )

    def test_analisis_y_dibujo_geometrico(self) -> None:
        mascara = np.zeros((64, 64), dtype=np.uint8)
        cv2.ellipse(mascara, (32, 32), (12, 22), 20, 0, 360, 255, thickness=-1)
        imagen = np.zeros((64, 64, 3), dtype=np.uint8)

        analisis = analizar_geometria(mascara)
        dibujada, analisis_dibujado = dibujar_analisis_geometrico(imagen, mascara)

        self.assertIsNotNone(analisis)
        self.assertIsNotNone(analisis_dibujado)
        self.assertGreater(int(dibujada.sum()), 0)
        self.assertGreater(analisis.solidez, 0)


if __name__ == "__main__":
    unittest.main()
