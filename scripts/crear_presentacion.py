from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


OUT = Path("docs/presentacion_logica_programa.pptx")

EMU = 914400
W = int(13.333333 * EMU)
H = int(7.5 * EMU)

COLOR = {
    "bright_snow": "F8F9FA",
    "platinum": "E9ECEF",
    "alabaster": "DEE2E6",
    "pale": "CED4DA",
    "pale2": "ADB5BD",
    "slate": "6C757D",
    "iron": "495057",
    "gunmetal": "343A40",
    "carbon": "212529",
    "white": "FFFFFF",
}


def emu(x: float) -> int:
    return int(x * EMU)


def text_runs(text: str, size: int = 24, color: str = "212529", bold: bool = False) -> str:
    b = "<a:b/>" if bold else ""
    return (
        f'<a:r><a:rPr lang="es-MX" sz="{size * 100}" dirty="0">{b}'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
        f"<a:t>{escape(text)}</a:t></a:r>"
    )


def paragraph(text: str, size: int = 24, color: str = "212529", bold: bool = False, bullet: bool = False) -> str:
    bullet_xml = '<a:buChar char="•"/>' if bullet else "<a:buNone/>"
    return f"<a:p><a:pPr>{bullet_xml}</a:pPr>{text_runs(text, size, color, bold)}<a:endParaRPr lang=\"es-MX\"/></a:p>"


def textbox(shape_id: int, x: float, y: float, w: float, h: float, paragraphs: list[str], fill: str | None = None) -> str:
    fill_xml = (
        f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
        if fill
        else "<a:noFill/>"
    )
    body = "".join(paragraphs)
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="Texto {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
    {fill_xml}
    <a:ln><a:noFill/></a:ln>
  </p:spPr>
  <p:txBody>
    <a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0"/>
    <a:lstStyle/>
    {body}
  </p:txBody>
</p:sp>
"""


def rect(shape_id: int, x: float, y: float, w: float, h: float, fill: str, line: str = "DEE2E6", radius: bool = True) -> str:
    geom = "roundRect" if radius else "rect"
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="Forma {shape_id}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>
    <a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>
    <a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>
    <a:ln w="10000"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>
  </p:spPr>
</p:sp>
"""


def line(shape_id: int, x1: float, y1: float, x2: float, y2: float, color: str = "ADB5BD") -> str:
    return f"""
<p:cxnSp>
  <p:nvCxnSpPr><p:cNvPr id="{shape_id}" name="Linea {shape_id}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
  <p:spPr>
    <a:xfrm><a:off x="{emu(x1)}" y="{emu(y1)}"/><a:ext cx="{emu(x2 - x1)}" cy="{emu(y2 - y1)}"/></a:xfrm>
    <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
    <a:ln w="18000"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>
  </p:spPr>
</p:cxnSp>
"""


def slide_xml(title: str, subtitle: str | None, shapes: list[str], number: int) -> str:
    header = [
        rect(2, 0, 0, 13.333, 0.16, COLOR["carbon"], COLOR["carbon"], False),
        textbox(3, 0.65, 0.45, 10.4, 0.55, [paragraph(title, 26, COLOR["carbon"], True)]),
    ]
    if subtitle:
        header.append(textbox(4, 0.65, 0.98, 11.5, 0.36, [paragraph(subtitle, 12, COLOR["slate"])]))
    footer = [
        line(90, 0.65, 7.05, 12.65, 7.05),
        textbox(91, 0.65, 7.15, 8, 0.22, [paragraph("Reconocimiento de señas con CNN desde cero", 8, COLOR["slate"])]),
        textbox(92, 12.1, 7.15, 0.5, 0.22, [paragraph(str(number), 8, COLOR["slate"])]),
    ]
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
       xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
       xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
    {''.join(header + shapes + footer)}
  </p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def bullet_box(start_id: int, x: float, y: float, w: float, h: float, title: str, bullets: list[str]) -> list[str]:
    ps = [paragraph(title, 18, COLOR["carbon"], True)]
    ps.extend(paragraph(b, 13, COLOR["iron"], False, True) for b in bullets)
    return [rect(start_id, x, y, w, h, COLOR["white"]), textbox(start_id + 1, x + 0.22, y + 0.22, w - 0.44, h - 0.35, ps)]


def flow_node(shape_id: int, x: float, y: float, text: str, fill: str = "FFFFFF") -> list[str]:
    return [
        rect(shape_id, x, y, 1.75, 0.72, fill, COLOR["pale"]),
        textbox(shape_id + 1, x + 0.12, y + 0.18, 1.5, 0.35, [paragraph(text, 10, COLOR["carbon"], True)]),
    ]


def build_slides() -> list[str]:
    slides = []
    slides.append(slide_xml(
        "Reconocimiento de señas con CNN",
        "Lógica del programa y flujo de procesamiento",
        [
            textbox(10, 0.78, 2.35, 8.0, 0.8, [paragraph("Sistema de visión por computadora para clasificar señas estáticas.", 24, COLOR["carbon"], True)]),
            textbox(11, 0.8, 3.25, 7.5, 0.8, [paragraph("El modelo se entrena desde cero con muestras capturadas por el usuario.", 16, COLOR["iron"])]),
            rect(12, 9.4, 1.9, 2.75, 2.35, COLOR["platinum"], COLOR["pale"]),
            textbox(13, 9.75, 2.25, 2.05, 1.55, [paragraph("HSV + Bordes", 20, COLOR["carbon"], True), paragraph("Entrada limpia para la CNN", 12, COLOR["slate"])]),
        ],
        1,
    ))

    slides.append(slide_xml(
        "Objetivo del proyecto",
        "Convertir una imagen de cámara en una predicción de seña",
        bullet_box(10, 0.8, 1.75, 5.6, 3.85, "Problema", [
            "La cámara captura fondo, ropa, rostro e iluminación variable.",
            "El modelo debe enfocarse en la forma de la mano.",
            "Se busca evitar modelos preentrenados.",
        ]) + bullet_box(20, 6.9, 1.75, 5.3, 3.85, "Solución", [
            "Recortar una zona central de trabajo.",
            "Separar piel con HSV.",
            "Usar máscara y bordes como entrada de la CNN.",
            "Aplicar aumentación de datos durante entrenamiento.",
        ]),
        2,
    ))

    nodes = []
    coords = [(0.75, 2.05, "Cámara"), (2.75, 2.05, "ROI"), (4.75, 2.05, "HSV"), (6.75, 2.05, "Máscara"), (8.75, 2.05, "Bordes"), (10.75, 2.05, "CNN")]
    for i, (x, y, t) in enumerate(coords):
        nodes += flow_node(10 + i * 3, x, y, t, COLOR["white"] if i % 2 == 0 else COLOR["platinum"])
        if i < len(coords) - 1:
            nodes.append(line(70 + i, x + 1.75, y + 0.36, coords[i + 1][0], y + 0.36))
    nodes += [
        line(80, 11.62, 2.77, 11.62, 4.2),
        rect(81, 10.75, 4.2, 1.75, 0.72, COLOR["carbon"], COLOR["carbon"]),
        textbox(82, 10.91, 4.38, 1.45, 0.35, [paragraph("Predicción", 10, COLOR["white"], True)]),
        textbox(83, 1.0, 5.35, 10.6, 0.55, [paragraph("La lógica central consiste en limpiar la imagen antes de clasificarla.", 16, COLOR["iron"])]),
    ]
    slides.append(slide_xml("Flujo general", "De la cámara a la predicción", nodes, 3))

    slides.append(slide_xml(
        "Captura de datos",
        "Construcción del corpus propio",
        bullet_box(10, 0.8, 1.7, 5.5, 4.0, "Entrada", [
            "El usuario coloca la mano en el recuadro central.",
            "Cada clase se guarda en una carpeta propia.",
            "Ejemplo: data/raw/A, data/raw/B.",
        ]) + bullet_box(20, 6.8, 1.7, 5.5, 4.0, "Control de calidad", [
            "La ventana muestra máscara HSV y entrada CNN.",
            "Solo se guarda si se detecta mano válida.",
            "Esto reduce muestras con ruido visual.",
        ]),
        4,
    ))

    slides.append(slide_xml(
        "Preprocesamiento",
        "Reducción del ruido visual",
        bullet_box(10, 0.75, 1.6, 3.7, 4.35, "HSV", [
            "Convierte BGR a HSV.",
            "Aísla tonos compatibles con piel.",
            "Permite calibración por iluminación.",
        ]) + bullet_box(20, 4.85, 1.6, 3.7, 4.35, "Morfología", [
            "Opening elimina puntos pequeños.",
            "Closing rellena huecos.",
            "GaussianBlur suaviza la máscara.",
        ]) + bullet_box(30, 8.95, 1.6, 3.7, 4.35, "Contorno", [
            "Busca el componente más grande.",
            "Rechaza áreas demasiado pequeñas.",
            "Extrae el ROI de la mano.",
        ]),
        5,
    ))

    slides.append(slide_xml(
        "Entrada de la CNN",
        "La red no recibe la foto completa",
        [
            rect(10, 1.0, 2.0, 3.0, 2.35, COLOR["platinum"], COLOR["pale"]),
            textbox(11, 1.35, 2.55, 2.3, 0.7, [paragraph("Canal 1", 18, COLOR["carbon"], True), paragraph("Máscara HSV", 13, COLOR["iron"])]),
            rect(12, 5.15, 2.0, 3.0, 2.35, COLOR["alabaster"], COLOR["pale"]),
            textbox(13, 5.5, 2.55, 2.3, 0.7, [paragraph("Canal 2", 18, COLOR["carbon"], True), paragraph("Bordes", 13, COLOR["iron"])]),
            rect(14, 9.3, 2.0, 2.7, 2.35, COLOR["carbon"], COLOR["carbon"]),
            textbox(15, 9.65, 2.55, 2.0, 0.9, [paragraph("Tensor", 18, COLOR["white"], True), paragraph("2 x 64 x 64", 13, COLOR["platinum"])]),
            line(16, 4.0, 3.18, 5.15, 3.18),
            line(17, 8.15, 3.18, 9.3, 3.18),
            textbox(18, 1.0, 5.25, 10.8, 0.6, [paragraph("Esta representación obliga al modelo a aprender forma y contorno, no fondo ni ropa.", 16, COLOR["iron"])]),
        ],
        6,
    ))

    slides.append(slide_xml(
        "Modelo CNN",
        "Red convolucional entrenada desde cero",
        bullet_box(10, 0.85, 1.65, 5.5, 4.1, "Arquitectura", [
            "Tres bloques convolucionales.",
            "BatchNorm + ReLU + MaxPool.",
            "Global Average Pooling.",
            "Capas densas con Dropout.",
        ]) + bullet_box(20, 6.85, 1.65, 5.4, 4.1, "Salida", [
            "Una neurona por clase.",
            "Softmax convierte salidas en probabilidades.",
            "La clase con mayor probabilidad es la predicción.",
        ]),
        7,
    ))

    slides.append(slide_xml(
        "Entrenamiento",
        "Aprendizaje con pocos datos mediante aumentación",
        bullet_box(10, 0.75, 1.55, 3.75, 4.45, "División", [
            "75% entrenamiento.",
            "25% prueba.",
            "Matriz de confusión final.",
        ]) + bullet_box(20, 4.8, 1.55, 3.75, 4.45, "Aumentación", [
            "Rotación leve.",
            "Zoom.",
            "Desplazamiento.",
            "Ruido.",
            "Erosión/dilatación.",
        ]) + bullet_box(30, 8.85, 1.55, 3.75, 4.45, "Resultado", [
            "Modelo sign_cnn.pt.",
            "Reporte de precisión.",
            "cnn_confusion_matrix.png.",
        ]),
        8,
    ))

    slides.append(slide_xml(
        "Interfaz en vivo",
        "Uso práctico del modelo entrenado",
        bullet_box(10, 0.85, 1.65, 5.45, 4.2, "Funciones", [
            "Predicción continua.",
            "Vista de máscara HSV.",
            "Calibración HSV con sliders.",
            "Muestra de piel desde el centro.",
        ]) + bullet_box(20, 6.85, 1.65, 5.45, 4.2, "Estabilidad", [
            "Historial de lecturas recientes.",
            "Umbral mínimo de confianza.",
            "Evita saltos por predicciones aisladas.",
        ]),
        9,
    ))

    slides.append(slide_xml(
        "Conclusión",
        "Resumen técnico",
        [
            textbox(10, 0.9, 1.75, 11.2, 0.8, [paragraph("El proyecto respeta el enfoque académico: procesamiento clásico de imágenes + CNN propia.", 22, COLOR["carbon"], True)]),
            *bullet_box(20, 1.1, 3.0, 10.8, 2.6, "Aportes principales", [
                "No depende de modelos preentrenados.",
                "Usa HSV, morfología, contornos y bordes.",
                "Entrena una CNN desde cero con aumentación de datos.",
                "Incluye evaluación e interfaz de calibración en tiempo real.",
            ]),
        ],
        10,
    ))
    return slides


def write_pptx(slides: list[str]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUT, "w", ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types(len(slides)))
        z.writestr("_rels/.rels", package_rels())
        z.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        z.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        z.writestr("ppt/slideMasters/slideMaster1.xml", slide_master())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels())
        z.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels())
        z.writestr("ppt/theme/theme1.xml", theme_xml())
        z.writestr("docProps/core.xml", core_xml())
        z.writestr("docProps/app.xml", app_xml(len(slides)))
        for i, s in enumerate(slides, 1):
            z.writestr(f"ppt/slides/slide{i}.xml", s)
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels())


def content_types(n: int) -> str:
    slides = "".join(f'<Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>' for i in range(1, n + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
<Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
{slides}</Types>"""


def package_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def presentation_xml(n: int) -> str:
    ids = "".join(f'<p:sldId id="{255+i}" r:id="rId{i}"/>' for i in range(1, n + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId{n+1}"/></p:sldMasterIdLst>
<p:sldIdLst>{ids}</p:sldIdLst>
<p:sldSz cx="{W}" cy="{H}" type="wide"/>
<p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def presentation_rels(n: int) -> str:
    rels = "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>' for i in range(1, n + 1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{rels}
<Relationship Id="rId{n+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
</Relationships>"""


def slide_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
</Relationships>"""


def slide_master() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
<p:sldLayoutIdLst><p:sldLayoutId id="1" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>"""


def master_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def slide_layout() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
<p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
</p:sldLayout>"""


def layout_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""


def theme_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="GrayscaleAcademic">
<a:themeElements><a:clrScheme name="Gray"><a:dk1><a:srgbClr val="212529"/></a:dk1><a:lt1><a:srgbClr val="F8F9FA"/></a:lt1><a:dk2><a:srgbClr val="343A40"/></a:dk2><a:lt2><a:srgbClr val="E9ECEF"/></a:lt2><a:accent1><a:srgbClr val="495057"/></a:accent1><a:accent2><a:srgbClr val="6C757D"/></a:accent2><a:accent3><a:srgbClr val="ADB5BD"/></a:accent3><a:accent4><a:srgbClr val="CED4DA"/></a:accent4><a:accent5><a:srgbClr val="DEE2E6"/></a:accent5><a:accent6><a:srgbClr val="E9ECEF"/></a:accent6><a:hlink><a:srgbClr val="495057"/></a:hlink><a:folHlink><a:srgbClr val="6C757D"/></a:folHlink></a:clrScheme><a:fontScheme name="Aptos"><a:majorFont><a:latin typeface="Aptos Display"/></a:majorFont><a:minorFont><a:latin typeface="Aptos"/></a:minorFont></a:fontScheme><a:fmtScheme name="Clean"><a:fillStyleLst><a:solidFill><a:schemeClr val="lt1"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="9525"><a:solidFill><a:schemeClr val="accent4"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="lt1"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme></a:themeElements></a:theme>"""


def core_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
<dc:title>Reconocimiento de señas con CNN</dc:title><dc:creator>Codex</dc:creator><cp:lastModifiedBy>Codex</cp:lastModifiedBy></cp:coreProperties>"""


def app_xml(n: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
<Application>Microsoft PowerPoint</Application><PresentationFormat>Widescreen</PresentationFormat><Slides>{n}</Slides></Properties>"""


if __name__ == "__main__":
    write_pptx(build_slides())
    print(OUT)
