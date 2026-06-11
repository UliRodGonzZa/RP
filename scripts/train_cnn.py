from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sign_recognition.cnn_model import (
    RedConvolucionalSenas,
    aumentar_entrada_cnn,
    crear_modelo_hibrido,
    entrada_cnn_desde_bgr,
    guardar_modelo_cnn,
)
from sign_recognition.dataset import dividir_indices_por_grupo, identificar_grupo_captura, iterar_imagenes
from sign_recognition.preprocess import ConfiguracionPreprocesamiento
from sign_recognition.shape_features import extraer_caracteristicas_forma


class DatasetSenasCNN(Dataset):
    """Dataset de PyTorch con aumentacion opcional."""

    def __init__(self, muestras: list[np.ndarray], objetivos: list[int], aumentar: bool = False) -> None:
        self.muestras = muestras
        self.objetivos = objetivos
        self.aumentar = aumentar

    def __len__(self) -> int:
        return len(self.muestras)

    def __getitem__(self, indice: int):
        muestra = self.muestras[indice]
        if self.aumentar:
            muestra = aumentar_entrada_cnn(muestra)
        caracteristicas_forma = extraer_caracteristicas_forma(muestra[0])
        return (
            torch.from_numpy(muestra),
            torch.from_numpy(caracteristicas_forma),
            torch.tensor(self.objetivos[indice], dtype=torch.long),
        )


def cargar_muestras_cnn(
    directorio_datos: Path,
    config: ConfiguracionPreprocesamiento,
    tamano_bloque: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, list[str]]:
    """Lee imagenes por carpeta y las convierte a entradas de 2 canales."""
    muestras: list[np.ndarray] = []
    etiquetas: list[str] = []
    grupos: list[str] = []
    estadisticas: dict[str, dict[str, int]] = {}

    for etiqueta, ruta_imagen in iterar_imagenes(directorio_datos):
        estadisticas.setdefault(etiqueta, {"total": 0, "usadas": 0, "omitidas": 0})
        estadisticas[etiqueta]["total"] += 1
        imagen = cv2.imread(str(ruta_imagen))
        if imagen is None:
            estadisticas[etiqueta]["omitidas"] += 1
            continue

        muestra = entrada_cnn_desde_bgr(imagen, config)
        if muestra is None:
            estadisticas[etiqueta]["omitidas"] += 1
            continue

        muestras.append(muestra)
        etiquetas.append(etiqueta)
        grupos.append(identificar_grupo_captura(directorio_datos, etiqueta, ruta_imagen, tamano_bloque))
        estadisticas[etiqueta]["usadas"] += 1

    print("Carga por clase:")
    print("Clase | Total | Usadas | Omitidas")
    for etiqueta in sorted(estadisticas):
        item = estadisticas[etiqueta]
        print(f"{etiqueta:>5} | {item['total']:>5} | {item['usadas']:>6} | {item['omitidas']:>8}")

    if not muestras:
        raise RuntimeError(f"No se pudieron cargar muestras validas desde {directorio_datos}")
    return muestras, np.array(etiquetas), np.array(grupos), sorted(set(etiquetas))


def evaluar(
    modelo: nn.Module,
    cargador: DataLoader,
    dispositivo: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    modelo.eval()
    correctas = 0
    total = 0
    reales: list[int] = []
    predichas: list[int] = []
    confianzas: list[float] = []
    margenes: list[float] = []
    with torch.no_grad():
        for x, forma, y in cargador:
            x = x.to(dispositivo)
            forma = forma.to(dispositivo)
            y = y.to(dispositivo)
            salidas = modelo(x, forma if modelo.numero_caracteristicas_forma else None)
            probabilidades = torch.softmax(salidas, dim=1)
            dos_mejores, indices_mejores = probabilidades.topk(k=2, dim=1)
            confianza = dos_mejores[:, 0]
            margen = dos_mejores[:, 0] - dos_mejores[:, 1]
            prediccion = indices_mejores[:, 0]
            correctas += int((prediccion == y).sum().item())
            total += int(y.numel())
            reales.extend(y.cpu().numpy().tolist())
            predichas.extend(prediccion.cpu().numpy().tolist())
            confianzas.extend(confianza.cpu().numpy().tolist())
            margenes.extend(margen.cpu().numpy().tolist())
    return (
        correctas / max(total, 1),
        np.array(reales),
        np.array(predichas),
        np.array(confianzas),
        np.array(margenes),
    )


def calibrar_umbrales_rechazo(
    reales: np.ndarray,
    predichas: np.ndarray,
    confianzas: np.ndarray,
    margenes: np.ndarray,
    precision_objetivo: float = 0.98,
    cobertura_minima: float = 0.50,
) -> tuple[float, float, float, float]:
    """Maximiza cobertura con una precision selectiva objetivo."""
    candidatos: list[tuple[float, float, float, float]] = []
    respaldo: list[tuple[float, float, float, float]] = []
    for umbral_confianza in np.arange(0.30, 0.951, 0.02):
        for umbral_margen in np.arange(0.02, 0.501, 0.02):
            aceptadas = (confianzas >= umbral_confianza) & (margenes >= umbral_margen)
            cobertura = float(aceptadas.mean())
            if cobertura < cobertura_minima:
                continue
            precision = float((reales[aceptadas] == predichas[aceptadas]).mean())
            candidato = (cobertura, precision, float(umbral_confianza), float(umbral_margen))
            respaldo.append(candidato)
            if precision >= precision_objetivo:
                candidatos.append(candidato)

    if candidatos:
        cobertura, precision, umbral_confianza, umbral_margen = max(candidatos)
    elif respaldo:
        cobertura, precision, umbral_confianza, umbral_margen = max(
            respaldo,
            key=lambda item: (item[1], item[0]),
        )
    else:
        return 0.50, 0.10, 0.0, 0.0

    return (
        round(umbral_confianza, 2),
        round(umbral_margen, 2),
        precision,
        cobertura,
    )


def guardar_curvas(historial: dict[str, list[float]], ruta: Path) -> None:
    epocas = np.arange(1, len(historial["perdida_entrenamiento"]) + 1)
    figura, ejes = plt.subplots(1, 2, figsize=(10, 4))
    ejes[0].plot(epocas, historial["perdida_entrenamiento"])
    ejes[0].set(title="Perdida de entrenamiento", xlabel="Epoca", ylabel="Perdida")
    ejes[1].plot(epocas, historial["exactitud_validacion"])
    ejes[1].set(title="Exactitud de validacion", xlabel="Epoca", ylabel="Exactitud", ylim=(0, 1))
    figura.tight_layout()
    figura.savefig(ruta, dpi=160)
    plt.close(figura)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena una CNN desde cero con mascara HSV y bordes.")
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--model-out", type=Path, default=Path("models/sign_cnn.pt"))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--edge-mode", choices=["canny", "sobel"], default="canny")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--capture-block-size", type=int, default=10)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-type", choices=["cnn", "hybrid"], default="hybrid")
    parser.add_argument("--target-selective-precision", type=float, default=0.98)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {dispositivo}")

    config = ConfiguracionPreprocesamiento(modo_bordes=args.edge_mode)
    muestras, etiquetas, grupos, clases = cargar_muestras_cnn(args.data, config, args.capture_block_size)
    indice_por_clase = {etiqueta: indice for indice, etiqueta in enumerate(clases)}
    objetivos = np.array([indice_por_clase[etiqueta] for etiqueta in etiquetas])

    indices_entrenamiento, indices_validacion, indices_prueba = dividir_indices_por_grupo(
        etiquetas,
        grupos,
        proporcion_validacion=args.val_size,
        proporcion_prueba=args.test_size,
        semilla=args.seed,
    )
    print(
        "Division por sesiones/bloques: "
        f"train={len(indices_entrenamiento)}, val={len(indices_validacion)}, test={len(indices_prueba)}"
    )

    muestras_entrenamiento = [muestras[i] for i in indices_entrenamiento]
    objetivos_entrenamiento = [int(objetivos[i]) for i in indices_entrenamiento]
    muestras_validacion = [muestras[i] for i in indices_validacion]
    objetivos_validacion = [int(objetivos[i]) for i in indices_validacion]
    muestras_prueba = [muestras[i] for i in indices_prueba]
    objetivos_prueba = [int(objetivos[i]) for i in indices_prueba]

    cargador_entrenamiento = DataLoader(
        DatasetSenasCNN(muestras_entrenamiento, objetivos_entrenamiento, aumentar=True),
        batch_size=args.batch_size,
        shuffle=True,
    )
    cargador_validacion = DataLoader(
        DatasetSenasCNN(muestras_validacion, objetivos_validacion, aumentar=False),
        batch_size=args.batch_size,
        shuffle=False,
    )
    cargador_prueba = DataLoader(
        DatasetSenasCNN(muestras_prueba, objetivos_prueba, aumentar=False),
        batch_size=args.batch_size,
        shuffle=False,
    )

    caracteristicas_entrenamiento = np.stack(
        [extraer_caracteristicas_forma(muestras[i][0]) for i in indices_entrenamiento]
    )
    media_forma = caracteristicas_entrenamiento.mean(axis=0).astype(np.float32)
    desviacion_forma = caracteristicas_entrenamiento.std(axis=0).astype(np.float32)
    desviacion_forma = np.maximum(desviacion_forma, 1e-6)

    if args.model_type == "hybrid":
        modelo = crear_modelo_hibrido(len(clases), media_forma, desviacion_forma).to(dispositivo)
    else:
        modelo = RedConvolucionalSenas(numero_clases=len(clases)).to(dispositivo)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=0.001)
    planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizador,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
    )
    funcion_perdida = nn.CrossEntropyLoss()
    mejor_exactitud = 0.0
    mejor_estado = None
    epocas_sin_mejora = 0
    historial = {"perdida_entrenamiento": [], "exactitud_validacion": []}

    for epoca in range(1, args.epochs + 1):
        modelo.train()
        perdida_acumulada = 0.0
        for x, forma, y in cargador_entrenamiento:
            x = x.to(dispositivo)
            forma = forma.to(dispositivo)
            y = y.to(dispositivo)
            optimizador.zero_grad()
            perdida = funcion_perdida(modelo(x, forma if modelo.numero_caracteristicas_forma else None), y)
            perdida.backward()
            optimizador.step()
            perdida_acumulada += float(perdida.item()) * int(y.numel())

        exactitud, _, _, _, _ = evaluar(modelo, cargador_validacion, dispositivo)
        perdida_promedio = perdida_acumulada / max(len(cargador_entrenamiento.dataset), 1)
        historial["perdida_entrenamiento"].append(perdida_promedio)
        historial["exactitud_validacion"].append(exactitud)
        planificador.step(exactitud)
        print(f"Epoca {epoca:03d}/{args.epochs} | perdida={perdida_promedio:.4f} | exactitud_val={exactitud:.4f}")

        if exactitud > mejor_exactitud:
            mejor_exactitud = exactitud
            mejor_estado = {clave: valor.detach().cpu().clone() for clave, valor in modelo.state_dict().items()}
            epocas_sin_mejora = 0
        else:
            epocas_sin_mejora += 1
            if epocas_sin_mejora >= args.patience:
                print(f"Early stopping tras {epoca} epocas.")
                break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)

    _, reales_val, predichas_val, confianzas_val, margenes_val = evaluar(
        modelo,
        cargador_validacion,
        dispositivo,
    )
    umbral_confianza, umbral_margen, precision_selectiva_val, cobertura_val = calibrar_umbrales_rechazo(
        reales_val,
        predichas_val,
        confianzas_val,
        margenes_val,
        precision_objetivo=args.target_selective_precision,
    )
    exactitud, reales, predichas, confianzas_prueba, margenes_prueba = evaluar(
        modelo,
        cargador_prueba,
        dispositivo,
    )
    aceptadas_prueba = (confianzas_prueba >= umbral_confianza) & (margenes_prueba >= umbral_margen)
    cobertura_prueba = float(aceptadas_prueba.mean())
    precision_selectiva_prueba = (
        float((reales[aceptadas_prueba] == predichas[aceptadas_prueba]).mean())
        if aceptadas_prueba.any()
        else 0.0
    )
    reporte = classification_report(
        reales,
        predichas,
        target_names=clases,
        zero_division=0,
        output_dict=True,
    )
    print(classification_report(reales, predichas, target_names=clases, zero_division=0))

    metadatos = {
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "semilla": args.seed,
        "epocas_ejecutadas": len(historial["perdida_entrenamiento"]),
        "mejor_exactitud_validacion": mejor_exactitud,
        "exactitud_prueba": exactitud,
        "umbral_confianza": umbral_confianza,
        "umbral_margen": umbral_margen,
        "modelo": args.model_type,
        "precision_selectiva_validacion": precision_selectiva_val,
        "cobertura_validacion": cobertura_val,
        "precision_selectiva_prueba": precision_selectiva_prueba,
        "cobertura_prueba": cobertura_prueba,
        "division": {
            "entrenamiento": len(indices_entrenamiento),
            "validacion": len(indices_validacion),
            "prueba": len(indices_prueba),
            "agrupada_por_captura": True,
        },
    }
    guardar_modelo_cnn(args.model_out, modelo.cpu(), clases, config, exactitud, metadatos)
    print(f"Modelo CNN guardado en: {args.model_out}")
    print(
        "Rechazo calibrado: "
        f"confianza>={umbral_confianza:.0%}, margen>={umbral_margen:.0%}, "
        f"precision_selectiva_test={precision_selectiva_prueba:.1%}, cobertura_test={cobertura_prueba:.1%}"
    )

    matriz = confusion_matrix(reales, predichas, labels=list(range(len(clases))))
    visualizador = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=clases)
    visualizador.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    ruta_matriz = args.model_out.parent / "cnn_confusion_matrix.png"
    plt.savefig(ruta_matriz, dpi=160)
    plt.close()
    print(f"Matriz de confusion guardada en: {ruta_matriz}")

    predichas_con_rechazo = np.array(clases, dtype=object)[predichas]
    predichas_con_rechazo[~aceptadas_prueba] = "NR"
    matriz_rechazo = confusion_matrix(
        np.array(clases, dtype=object)[reales],
        predichas_con_rechazo,
        labels=[*clases, "NR"],
    )
    visualizador_rechazo = ConfusionMatrixDisplay(
        confusion_matrix=matriz_rechazo,
        display_labels=[*clases, "NR"],
    )
    visualizador_rechazo.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    ruta_matriz_rechazo = args.model_out.parent / "cnn_rejection_matrix.png"
    plt.savefig(ruta_matriz_rechazo, dpi=160)
    plt.close()
    print(f"Matriz con rechazo guardada en: {ruta_matriz_rechazo}")

    ruta_curvas = args.model_out.parent / "cnn_training_curves.png"
    guardar_curvas(historial, ruta_curvas)
    ruta_metricas = args.model_out.parent / "cnn_metrics.json"
    ruta_metricas.write_text(
        json.dumps({"metadatos": metadatos, "reporte_clasificacion": reporte}, indent=2),
        encoding="utf-8",
    )
    print(f"Metricas guardadas en: {ruta_metricas}")


if __name__ == "__main__":
    main()
