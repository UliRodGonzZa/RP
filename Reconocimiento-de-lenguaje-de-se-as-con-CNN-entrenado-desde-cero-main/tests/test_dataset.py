import unittest

import numpy as np

from sign_recognition.dataset import dividir_indices_por_grupo


class DivisionAgrupadaTests(unittest.TestCase):
    def test_no_mezcla_grupos_entre_conjuntos(self) -> None:
        etiquetas = np.array(["A"] * 12 + ["B"] * 12)
        grupos = np.array(
            [f"A/sesion_{indice // 2}" for indice in range(12)]
            + [f"B/sesion_{indice // 2}" for indice in range(12)]
        )

        train, validacion, prueba = dividir_indices_por_grupo(
            etiquetas,
            grupos,
            proporcion_validacion=0.2,
            proporcion_prueba=0.2,
            semilla=7,
        )

        grupos_train = set(grupos[train])
        grupos_validacion = set(grupos[validacion])
        grupos_prueba = set(grupos[prueba])
        self.assertFalse(grupos_train & grupos_validacion)
        self.assertFalse(grupos_train & grupos_prueba)
        self.assertFalse(grupos_validacion & grupos_prueba)

        for indices in (train, validacion, prueba):
            self.assertEqual(set(etiquetas[indices]), {"A", "B"})

    def test_requiere_tres_grupos_por_clase(self) -> None:
        with self.assertRaises(ValueError):
            dividir_indices_por_grupo(
                np.array(["A", "A"]),
                np.array(["A/s1", "A/s2"]),
            )


if __name__ == "__main__":
    unittest.main()
