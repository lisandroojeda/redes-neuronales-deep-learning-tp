# Segmentación semántica de césped con CSIRO Biomass Seg

Este repositorio reúne la implementación reproducible y el informe de un proyecto de segmentación semántica de fotografías de pasturas. El objetivo es clasificar cada píxel válido como césped o vegetación verde, suelo expuesto u otra vegetación. La salida de suelo se interpreta como un sector candidato a revisión, no como una recomendación agronómica.

## Estado del proyecto

La estructura del dataset fue verificada con el archivo real: 357 imágenes, 357 máscaras, ningún par faltante, tamaños alineados y valores de máscara entre 0 y 5. La corrida completa se ejecutó en Google Colab con GPU. El mejor experimento fue E3, con mean IoU 0,6976 y macro Dice 0,8095 en test.

El notebook principal se agrega en `notebooks/Proyecto_Cesped_CSIRO.ipynb` antes de publicar el repositorio.

## Dataset

Se utiliza [CSIRO-Biomass-Seg](https://www.kaggle.com/datasets/qiyuliaocsiro/csiro-biomass-seg), publicado en Kaggle con licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

El dataset es público y puede compartirse respetando esa licencia. La descarga verificada contiene 357 fotografías RGB y 357 máscaras semánticas correspondientes, todas de 2000 × 1000 píxeles. Cada imagen tiene una única máscara asociada; no se encontraron pares faltantes ni duplicados exactos. Las máscaras son monocanal y utilizan valores enteros de 0 a 5.

Codificación original documentada:

| ID | Clase original | Clase utilizada |
|---:|---|---|
| 0 | Background | Ignorar 255 |
| 1 | Clover | Otra vegetación 2 |
| 2 | Green | Césped o vegetación verde 0 |
| 3 | Dead Matter | Ignorar 255 |
| 4 | Ground | Suelo expuesto 1 |
| 5 | Covering | Ignorar 255 |

`Green` se usa como aproximación de césped. El dataset no distingue especies de gramíneas ni representa necesariamente jardines residenciales.

## Requisitos

- Una cuenta de Google para usar Colab y Drive.
- Acceso a Kaggle para descargar el dataset público.
- Un runtime de Google Colab con GPU.
- El notebook `Proyecto_Cesped_CSIRO.ipynb`, que debe incorporarse en `notebooks/` antes de publicar el repositorio.

## Reproducción en Google Colab

1. Descargá el ZIP desde [CSIRO-Biomass-Seg en Kaggle](https://www.kaggle.com/datasets/qiyuliaocsiro/csiro-biomass-seg). Kaggle puede solicitar que inicies sesión.
2. Renombralo como `csiro-biomass-seg.zip`.
3. Subilo a `MyDrive/Proyecto_Cesped/csiro-biomass-seg.zip`.
4. Subí `Proyecto_Cesped_CSIRO.ipynb` a Colab o abrilo desde Google Drive.
5. Seleccioná **Entorno de ejecución > Cambiar tipo de entorno de ejecución > GPU**.
6. Ejecutá **Entorno de ejecución > Ejecutar todas**. No omitas las validaciones iniciales: comprueban integridad del ZIP, correspondencia entre imágenes y máscaras, dimensiones y valores de clase.
7. Esperá a que finalicen la prueba de humo y los experimentos E0–E3. E4 y E5 sólo se ejecutan si se proporciona un dataset local anotado.
8. Descargá `/content/Proyecto_Cesped/outputs/` o copialo a `results/` para conservar `versions.json`, manifiestos, historiales, checkpoints, figuras y `results_summary.csv`.

La corrida documentada hace primero una prueba de humo y luego usa 300 imágenes: 210 para entrenamiento, 45 para validación y 45 para test. Los duplicados exactos y los sitios conocidos se mantienen dentro de una sola partición. Validation selecciona hiperparámetros y checkpoints; test se consulta únicamente después de elegir el modelo.

## Experimentos

- E0: baseline binario HSV, Excess Green y morfología.
- E1: U-Net con ResNet-34 entrenada desde cero.
- E2: U-Net con encoder ResNet-34 preentrenado en ImageNet y congelado.
- E3: fine-tuning completo desde el mejor checkpoint de E2.
- E4: evaluación directa opcional en fotografías locales.
- E5: adaptación opcional con datos locales anotados.

El checkpoint se selecciona por mean IoU de validación. Test se reserva para la evaluación final. También se informan IoU y Dice por clase, macro Dice, precisión, recall y F1 para suelo, y error absoluto de los porcentajes por clase.

## Dependencias y entorno

Las dependencias instaladas por el notebook están fijadas en `requirements-colab.txt`. PyTorch y torchvision se toman del runtime de Colab para conservar compatibilidad con CUDA. Sus versiones efectivas, junto con Python, CUDA, NumPy y pandas, se guardan automáticamente en `results/versions.json`.

Versiones efectivas de la corrida utilizada en el informe:

| Componente | Versión |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu130 |
| CUDA runtime | 13.0 |
| segmentation-models-pytorch | 0.5.0 |
| Albumentations | 1.4.24 |
| OpenCV | 4.14.0 |
| NumPy | 2.1.3 |
| pandas | 2.2.3 |
| seaborn | 0.13.2 |
| scikit-learn | 1.7.2 |
| tabulate | 0.9.0 |

El notebook instala `segmentation-models-pytorch==0.5.0`, `albumentations==1.4.24`, `seaborn==0.13.2`, `tabulate==0.9.0` y `scikit-learn>=1.6,<1.8`. PyTorch, torchvision y OpenCV se conservan desde el runtime de Colab para evitar incompatibilidades con CUDA. La versión exacta del entorno siempre debe tomarse de `versions.json`, porque Colab puede actualizar sus paquetes base.

## Semillas y determinismo

La única semilla global fijada es `42`. Se aplica a:

- `random` de Python.
- NumPy.
- PyTorch en CPU.
- PyTorch en todas las GPU mediante `torch.cuda.manual_seed_all`.
- Generadores y procesos auxiliares de cada `DataLoader`.
- Selección reproducible del subconjunto y partición train/validation/test.

Además se configura:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

La selección de las 300 imágenes se ordena mediante SHA-256 combinado con la semilla. La partición también usa la misma semilla. Aun así, operaciones CUDA o cambios del runtime pueden introducir pequeñas diferencias numéricas; por eso la corrida registra las versiones efectivas.

## Resultados esperados de la reproducción

La reproducción puede presentar pequeñas variaciones numéricas, pero debería mantener el mismo comportamiento general: E2 converge más rápido y supera a E1, mientras E3 obtiene el mejor checkpoint. En la corrida documentada E3 alcanzó mean IoU 0,6976, macro Dice 0,8095 y F1 de suelo 0,8557 en test. No se debe ajustar ningún parámetro a partir del conjunto de test.

## Informe

El informe está disponible en formato PDF dentro de `docs/`. Documenta las decisiones, dificultades, aprendizajes, curvas y resultados de la corrida completa.

## Alcance

El modelo produce proporciones de píxeles válidos y mapas visuales. No estima metros cuadrados, biomasa ni necesidades de tratamiento. Antes de usarlo sobre jardines o imágenes de celular, hace falta validar el cambio de dominio con fotografías locales representativas.
