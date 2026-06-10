from __future__ import annotations

import argparse
import sys
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sign_recognition.camera import abrir_camara
from sign_recognition.cnn_model import cargar_metadatos_modelo, cargar_modelo_cnn
from sign_recognition.preprocess import (
    RangoHSV,
    crear_mascara_piel,
    dibujar_caja_mano,
    dibujar_roi_central,
    limites_roi_central,
    preprocesar_imagen,
    recortar_roi_central,
)
from sign_recognition.shape_features import dibujar_analisis_geometrico, extraer_caracteristicas_forma


ESTILO_APP = """
QMainWindow {
    background: #f8f9fa;
    color: #212529;
}
QMenuBar {
    background: #f8f9fa;
    color: #343a40;
    border-bottom: 1px solid #dee2e6;
    padding: 6px 12px;
    font-size: 14px;
}
QMenuBar::item {
    padding: 7px 12px;
    border-radius: 6px;
}
QMenuBar::item:selected {
    background: #e9ecef;
}
QMenu {
    background: #ffffff;
    color: #212529;
    border: 1px solid #dee2e6;
    padding: 6px;
}
QMenu::item {
    padding: 8px 26px;
    border-radius: 6px;
}
QMenu::item:selected {
    background: #e9ecef;
}
QFrame#Encabezado, QFrame#PanelLateral, QFrame#BarraControles {
    background: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 10px;
}
QLabel#Titulo {
    color: #212529;
    font-size: 26px;
    font-weight: 700;
}
QLabel#Subtitulo, QLabel#TextoSecundario {
    color: #6c757d;
    font-size: 14px;
}
QLabel#Video {
    background: #212529;
    color: #f8f9fa;
    border: 1px solid #ced4da;
    border-radius: 10px;
}
QLabel#Prediccion {
    color: #212529;
    font-size: 42px;
    font-weight: 800;
}
QLabel#EtiquetaPanel {
    color: #495057;
    font-size: 13px;
    font-weight: 700;
}
QLabel#ValorPanel {
    color: #212529;
    font-size: 18px;
    font-weight: 600;
}
QPushButton {
    background: #ffffff;
    color: #343a40;
    border: 1px solid #ced4da;
    border-radius: 8px;
    padding: 9px 15px;
    font-size: 14px;
    font-weight: 600;
}
QPushButton:hover {
    background: #e9ecef;
    border-color: #adb5bd;
}
QPushButton:pressed {
    background: #dee2e6;
}
QCheckBox {
    color: #343a40;
    font-size: 14px;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #adb5bd;
    border-radius: 5px;
    background: #ffffff;
}
QCheckBox::indicator:checked {
    background: #495057;
    border-color: #495057;
}
QDialog {
    background: #f8f9fa;
    color: #212529;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #dee2e6;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #495057;
    width: 18px;
    height: 18px;
    margin: -7px 0;
    border-radius: 9px;
}
QSlider::sub-page:horizontal {
    background: #6c757d;
    border-radius: 2px;
}
"""


def convertir_frame_a_pixmap(frame_bgr: np.ndarray, ancho_max: int = 900, alto_max: int = 560) -> QPixmap:
    """Convierte un frame OpenCV BGR a imagen Qt escalada."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    alto, ancho, canales = rgb.shape
    imagen = QImage(rgb.data, ancho, alto, canales * ancho, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(imagen).scaled(ancho_max, alto_max, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def crear_linea_panel(titulo: str, valor_inicial: str) -> tuple[QWidget, QLabel]:
    contenedor = QWidget()
    layout = QVBoxLayout(contenedor)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(2)

    etiqueta = QLabel(titulo)
    etiqueta.setObjectName("EtiquetaPanel")
    valor = QLabel(valor_inicial)
    valor.setObjectName("ValorPanel")
    valor.setWordWrap(True)

    layout.addWidget(etiqueta)
    layout.addWidget(valor)
    return contenedor, valor


class DialogoCalibracion(QDialog):
    """Ventana para ajustar los limites HSV segun piel e iluminacion."""

    def __init__(self, ventana: "VentanaReconocimiento") -> None:
        super().__init__(ventana)
        self.ventana = ventana
        self.setWindowTitle("Calibracion HSV")
        self.setMinimumWidth(460)
        self.setStyleSheet(ESTILO_APP)

        self.sliders: dict[str, QSlider] = {}
        layout = QVBoxLayout(self)
        formulario = QFormLayout()

        valores = {
            "H minimo": ventana.config.rango_hsv.inferior[0],
            "S minimo": ventana.config.rango_hsv.inferior[1],
            "V minimo": ventana.config.rango_hsv.inferior[2],
            "H maximo": ventana.config.rango_hsv.superior[0],
            "S maximo": ventana.config.rango_hsv.superior[1],
            "V maximo": ventana.config.rango_hsv.superior[2],
        }
        limites = {"H minimo": 179, "H maximo": 179, "S minimo": 255, "S maximo": 255, "V minimo": 255, "V maximo": 255}

        for nombre, valor in valores.items():
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, limites[nombre])
            slider.setValue(valor)
            slider.valueChanged.connect(self.aplicar_valores)
            self.sliders[nombre] = slider
            formulario.addRow(nombre, slider)

        boton_muestra = QPushButton("Tomar muestra de piel del centro")
        boton_muestra.clicked.connect(self.tomar_muestra_piel)

        boton_cerrar = QPushButton("Cerrar")
        boton_cerrar.clicked.connect(self.accept)

        layout.addLayout(formulario)
        layout.addWidget(boton_muestra)
        layout.addWidget(boton_cerrar)

    def valores_hsv(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        inferior = (
            self.sliders["H minimo"].value(),
            self.sliders["S minimo"].value(),
            self.sliders["V minimo"].value(),
        )
        superior = (
            self.sliders["H maximo"].value(),
            self.sliders["S maximo"].value(),
            self.sliders["V maximo"].value(),
        )
        return inferior, superior

    def aplicar_valores(self) -> None:
        inferior, superior = self.valores_hsv()
        inferior_fijo = tuple(min(a, b) for a, b in zip(inferior, superior))
        superior_fijo = tuple(max(a, b) for a, b in zip(inferior, superior))
        self.ventana.actualizar_hsv(inferior_fijo, superior_fijo)

    def tomar_muestra_piel(self) -> None:
        if self.ventana.roi_actual is None:
            return

        roi = self.ventana.roi_actual
        alto, ancho = roi.shape[:2]
        muestra = roi[alto // 2 - 35 : alto // 2 + 35, ancho // 2 - 35 : ancho // 2 + 35]
        hsv = cv2.cvtColor(muestra, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        inferior = np.percentile(hsv, 8, axis=0).astype(int)
        superior = np.percentile(hsv, 92, axis=0).astype(int)
        margen = np.array([8, 35, 35])
        inferior = np.clip(inferior - margen, [0, 0, 0], [179, 255, 255])
        superior = np.clip(superior + margen, [0, 0, 0], [179, 255, 255])

        for nombre, valor in zip(
            ["H minimo", "S minimo", "V minimo", "H maximo", "S maximo", "V maximo"],
            [*inferior, *superior],
        ):
            self.sliders[nombre].blockSignals(True)
            self.sliders[nombre].setValue(int(valor))
            self.sliders[nombre].blockSignals(False)
        self.aplicar_valores()


class VentanaReconocimiento(QMainWindow):
    """Interfaz principal: camara, calibracion y prediccion CNN en tiempo real."""

    def __init__(self, ruta_modelo_cnn: Path, indice_camara: int, backend_camara: str) -> None:
        super().__init__()
        self.setWindowTitle("Reconocimiento de senas - CNN")
        self.resize(1200, 780)
        self.setStyleSheet(ESTILO_APP)

        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo_cnn, self.etiquetas, self.config = cargar_modelo_cnn(ruta_modelo_cnn, self.dispositivo)
        self.metadatos_modelo = cargar_metadatos_modelo(ruta_modelo_cnn)
        self.umbral_confianza = float(self.metadatos_modelo.get("umbral_confianza", 0.50))
        self.umbral_margen = float(self.metadatos_modelo.get("umbral_margen", 0.0))
        self.indice_camara = indice_camara
        self.backend_camara = backend_camara
        self.roi_actual: np.ndarray | None = None
        self.historial_predicciones: deque[str] = deque(maxlen=10)
        self.ultima_prediccion_estable = "sin lectura"
        self.intentos_reconexion = 0

        camara = abrir_camara(indice_camara, backend_camara)
        self.captura = camara.captura
        self.backend_activo = camara.nombre_backend

        self.construir_interfaz()
        self.construir_menu()

        self.temporizador = QTimer(self)
        self.temporizador.timeout.connect(self.actualizar_video)
        self.temporizador.start(33)

    def construir_interfaz(self) -> None:
        encabezado = QFrame()
        encabezado.setObjectName("Encabezado")
        encabezado_layout = QVBoxLayout(encabezado)
        encabezado_layout.setContentsMargins(18, 14, 18, 14)
        titulo = QLabel("Reconocimiento de senas con CNN")
        titulo.setObjectName("Titulo")
        subtitulo = QLabel("Modelo entrenado desde cero con mascara HSV y bordes. Coloca la mano dentro del recuadro.")
        subtitulo.setObjectName("Subtitulo")
        encabezado_layout.addWidget(titulo)
        encabezado_layout.addWidget(subtitulo)

        self.video = QLabel(alignment=Qt.AlignCenter)
        self.video.setObjectName("Video")
        self.video.setMinimumSize(850, 540)
        self.video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        panel = QFrame()
        panel.setObjectName("PanelLateral")
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(18, 18, 18, 18)
        panel_layout.setSpacing(18)

        texto_pred = QLabel("Prediccion")
        texto_pred.setObjectName("EtiquetaPanel")
        self.prediccion = QLabel("sin lectura")
        self.prediccion.setObjectName("Prediccion")

        bloque_confianza, self.confianza = crear_linea_panel("Confianza", "-")
        bloque_modelo, self.modelo_info = crear_linea_panel(
            "Modelo",
            (
                f"{self.metadatos_modelo.get('modelo', 'cnn').upper()} | {len(self.etiquetas)} clases | "
                f"conf. {self.umbral_confianza:.0%} | margen {self.umbral_margen:.0%}"
            ),
        )
        bloque_camara, self.camara_info = crear_linea_panel("Camara", self.backend_activo)
        bloque_hsv, self.hsv_info = crear_linea_panel("HSV", self.texto_hsv())
        bloque_geometria, self.geometria_info = crear_linea_panel("Geometria", "-")
        bloque_historial, self.historial_info = crear_linea_panel("Historial", "-")

        panel_layout.addWidget(texto_pred)
        panel_layout.addWidget(self.prediccion)
        panel_layout.addWidget(bloque_confianza)
        panel_layout.addWidget(bloque_modelo)
        panel_layout.addWidget(bloque_camara)
        panel_layout.addWidget(bloque_hsv)
        panel_layout.addWidget(bloque_geometria)
        panel_layout.addWidget(bloque_historial)
        panel_layout.addStretch(1)

        self.prediccion_continua = QCheckBox("Prediccion continua")
        self.prediccion_continua.setChecked(True)
        self.ver_mascara = QCheckBox("Ver mascara HSV")
        self.ver_geometria = QCheckBox("Ver geometria")

        boton_predecir = QPushButton("Predecir ahora")
        boton_predecir.clicked.connect(self.predecir)
        boton_calibrar = QPushButton("Calibrar HSV")
        boton_calibrar.clicked.connect(self.abrir_dialogo_calibracion)
        boton_reintentar = QPushButton("Reintentar camara")
        boton_reintentar.clicked.connect(self.reintentar_camara)

        barra = QFrame()
        barra.setObjectName("BarraControles")
        barra_layout = QHBoxLayout(barra)
        barra_layout.setContentsMargins(14, 10, 14, 10)
        barra_layout.setSpacing(12)
        barra_layout.addWidget(self.prediccion_continua)
        barra_layout.addWidget(self.ver_mascara)
        barra_layout.addWidget(self.ver_geometria)
        barra_layout.addStretch(1)
        barra_layout.addWidget(boton_predecir)
        barra_layout.addWidget(boton_calibrar)
        barra_layout.addWidget(boton_reintentar)

        contenido = QGridLayout()
        contenido.setColumnStretch(0, 4)
        contenido.setColumnStretch(1, 1)
        contenido.setSpacing(16)
        contenido.addWidget(self.video, 0, 0)
        contenido.addWidget(panel, 0, 1)

        raiz = QVBoxLayout()
        raiz.setContentsMargins(24, 18, 24, 24)
        raiz.setSpacing(14)
        raiz.addWidget(encabezado)
        raiz.addLayout(contenido)
        raiz.addWidget(barra)

        contenedor = QWidget()
        contenedor.setLayout(raiz)
        self.setCentralWidget(contenedor)

    def construir_menu(self) -> None:
        menu_calibracion = self.menuBar().addMenu("Calibracion")
        accion_ajustar = QAction("Ajustar HSV", self)
        accion_ajustar.triggered.connect(self.abrir_dialogo_calibracion)
        accion_restaurar = QAction("Restaurar HSV inicial", self)
        accion_restaurar.triggered.connect(self.restaurar_hsv)
        menu_calibracion.addAction(accion_ajustar)
        menu_calibracion.addAction(accion_restaurar)

    def texto_hsv(self) -> str:
        return f"{self.config.rango_hsv.inferior} - {self.config.rango_hsv.superior}"

    def actualizar_hsv(self, inferior: tuple[int, int, int], superior: tuple[int, int, int]) -> None:
        self.config = replace(self.config, rango_hsv=RangoHSV(inferior=inferior, superior=superior))
        self.hsv_info.setText(self.texto_hsv())

    def restaurar_hsv(self) -> None:
        self.actualizar_hsv((0, 25, 45), (25, 255, 255))

    def abrir_dialogo_calibracion(self) -> None:
        DialogoCalibracion(self).exec()

    def actualizar_video(self) -> None:
        lectura_correcta, frame = self.captura.read()
        if not lectura_correcta or frame is None:
            self.intentos_reconexion += 1
            if self.intentos_reconexion >= 20:
                self.reintentar_camara()
            return

        self.intentos_reconexion = 0
        frame = frame.copy()
        self.roi_actual = recortar_roi_central(frame.copy())

        vista = dibujar_roi_central(frame.copy())
        x0, y0, x1, y1 = limites_roi_central(vista)
        roi_vista = dibujar_caja_mano(vista[y0:y1, x0:x1].copy(), self.config)
        mascara = crear_mascara_piel(self.roi_actual, self.config.rango_hsv)

        if self.ver_mascara.isChecked():
            roi_vista = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)

        if self.ver_geometria.isChecked():
            roi_vista, analisis = dibujar_analisis_geometrico(roi_vista, mascara)
            if analisis is None:
                self.geometria_info.setText("sin silueta")
            else:
                self.geometria_info.setText(
                    f"Aspecto {analisis.relacion_aspecto:.2f} | "
                    f"Extension {analisis.extension:.2f}\n"
                    f"Solidez {analisis.solidez:.2f} | "
                    f"Circularidad {analisis.circularidad:.2f}"
                )
        else:
            self.geometria_info.setText("-")

        vista[y0:y1, x0:x1] = cv2.resize(roi_vista, (x1 - x0, y1 - y0))

        self.video.setPixmap(convertir_frame_a_pixmap(vista))

        if self.prediccion_continua.isChecked():
            self.predecir(silencioso=True)

    def predecir(self, silencioso: bool = False) -> None:
        if self.roi_actual is None:
            return

        procesada = preprocesar_imagen(self.roi_actual.copy(), self.config)
        if procesada is None:
            if not silencioso:
                self.prediccion.setText("sin mano")
            return

        bordes, mascara = procesada
        etiqueta, confianza, margen = self.predecir_con_cnn(mascara, bordes)
        if confianza < self.umbral_confianza or margen < self.umbral_margen:
            self.historial_predicciones.clear()
            self.prediccion.setText("no reconocida")
            self.confianza.setText(f"{confianza:.1%} | margen {margen:.1%}")
            self.historial_info.setText("-")
            return

        self.historial_predicciones.append(etiqueta)
        etiqueta_estable, cantidad = Counter(self.historial_predicciones).most_common(1)[0]
        if cantidad >= 4:
            self.ultima_prediccion_estable = etiqueta_estable

        self.prediccion.setText(self.ultima_prediccion_estable)
        self.confianza.setText(f"{confianza:.1%} | margen {margen:.1%}")
        self.historial_info.setText(", ".join(self.historial_predicciones))

    def predecir_con_cnn(self, mascara: np.ndarray, bordes: np.ndarray) -> tuple[str, float, float]:
        entrada = np.stack([mascara, bordes], axis=0).astype("float32") / 255.0
        tensor = torch.from_numpy(entrada).unsqueeze(0).to(self.dispositivo)
        tensor_forma = None
        if self.modelo_cnn.numero_caracteristicas_forma:
            caracteristicas = extraer_caracteristicas_forma(mascara)
            tensor_forma = torch.from_numpy(caracteristicas).unsqueeze(0).to(self.dispositivo)
        with torch.inference_mode():
            probabilidades = torch.softmax(self.modelo_cnn(tensor, tensor_forma), dim=1)[0]
        dos_mejores, indices = probabilidades.topk(k=2)
        confianza = float(dos_mejores[0].item())
        margen = float((dos_mejores[0] - dos_mejores[1]).item())
        return self.etiquetas[int(indices[0].item())], confianza, margen

    def reintentar_camara(self) -> None:
        try:
            self.captura.release()
        except Exception:
            pass
        try:
            camara = abrir_camara(self.indice_camara, self.backend_camara)
        except RuntimeError:
            self.video.setText("Camara no disponible. Revisa permisos o conexion.")
            self.camara_info.setText("sin frames")
            return

        self.captura = camara.captura
        self.backend_activo = camara.nombre_backend
        self.camara_info.setText(self.backend_activo)

    def closeEvent(self, evento) -> None:
        self.temporizador.stop()
        try:
            self.captura.release()
        except Exception:
            pass
        evento.accept()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interfaz CNN para reconocer senas en vivo.")
    parser.add_argument("--cnn-model", type=Path, default=Path("models/sign_cnn.pt"), help="Ruta al modelo CNN .pt.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"], default="auto")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    try:
        ventana = VentanaReconocimiento(args.cnn_model, args.camera, args.backend)
    except Exception as exc:
        QMessageBox.critical(None, "Error", str(exc))
        raise

    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
