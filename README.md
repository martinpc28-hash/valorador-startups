# Valorador de Startups

Aplicación web (Streamlit) que valora una startup a partir de las variables que introduce el usuario. Ancla sus supuestos en datos sectoriales con procedencia verificable. Combina cuatro métodos (DCF, método VC, múltiplos comparables y Monte Carlo), los compara con la pre-money propuesta y muestra de dónde viene cada número.

> Herramienta educativa y de análisis. **No constituye asesoramiento de inversión.**

<!-- Capturas: añadir en docs/ y enlazar aquí -->
<!-- ![Resumen](docs/resumen.png) -->

## Qué hace

| Pestaña | Contenido |
|---|---|
| **Resumen** | Rango de valor por método frente a la pre-money propuesta, con un veredicto (por debajo, dentro o por encima del rango) |
| **DCF** | Beta de la industria reapalancada (de mercado o total) y costo de capital. Ingresos que convergen a crecimiento estable, margen que converge al de la industria, reinversión con ventas / capital, pérdidas fiscales acumuladas, valor terminal y ajuste por supervivencia. Mapa de sensibilidad y gráfico de tornado |
| **Método VC** | Valor de salida (EV/Sales o EV/EBITDA) descontado a la IRR objetivo con dilución futura. Participación necesaria y MOIC/IRR con las condiciones propuestas |
| **Múltiplos** | EV/Sales y EV/EBITDA de la industria y de la clase de capitalización más pequeña, con descuento por iliquidez |
| **Escenarios** | Pesimista, base y optimista con factores editables sobre crecimiento, margen y múltiplo |
| **Monte Carlo** | 10.000 simulaciones con semilla fija. Crecimiento, margen y múltiplo correlacionados; fracaso simulado; P10/P50/P90 y probabilidad de alcanzar un MOIC objetivo |
| **Caja y ronda** | Runway, caja mensual, capital que consume el plan y dilución adicional implícita |
| **Comparables SEC** | Buscador de empresas reales en Form D (rondas privadas), Form C (startups con estados financieros) y S-1 (salidas a bolsa). Filtros por nombre, industria y antigüedad. Al seleccionar una o varias se comparan con tu startup y se ve en qué percentil queda tu ronda o tus ingresos. Un botón **precarga** los datos de la empresa elegida en el modelo |
| **Mis empresas** | Biblioteca **compartida y sin contraseña** (por ahora todos los visitantes ven las mismas empresas; no guardes datos confidenciales). Creas plantillas de empresas y subes sus estados financieros en Excel, CSV o PDF. La app detecta las partidas en español o inglés (ingresos, EBIT, beneficio neto, caja, deuda, flujo operativo, empleados) y la escala, y rellena ingresos, crecimiento, margen, caja, deuda, burn estimado y pérdidas acumuladas. Revisas, guardas, y desde la barra lateral la cargas en el modelo o la añades a la comparación |
| **Fondos** | Métricas de un fondo de VC (DPI, RVPI, TVPI, MOIC e IRR con XIRR propio), curva J, proyección tipo Takahashi-Alexander y tamaño frente a los vehículos de VC que presentaron Form D. Flujos editables o cargados desde CSV |
| **Supuestos** | Tabla editable por etapa: IRR objetivo, supervivencia, dilución, iliquidez |
| **Datos y fuentes** | Cada número usado con su fuente, fecha, URL y avisos (reemplazos y recortes) |

Decisiones de diseño que vale la pena señalar:

- **El fracaso no se cuenta dos veces.** En el método VC el usuario elige entre una IRR alta (que ya incluye el riesgo de fracaso) o el costo del equity × probabilidad de supervivencia. Nunca se aplican las dos.
- **La ronda es coherente por construcción.** Se introducen dos de inversión, pre-money y participación, y la app calcula la tercera.
- **Las tasas mensuales se componen.** La conversión es (1 + g)^(1/12) − 1, no g / 12.
- **No se promedian fuentes en silencio.** Si varias fuentes tienen la misma métrica, la app toma la primera del orden de prioridad, muestra las demás y avisa cuando hay reemplazo.

## Ejecutar en local

Requiere Python 3.11 o superior. La app no necesita internet: todos los datos están en `data/`.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (en macOS/Linux: source .venv/bin/activate)
pip install -r requirements-dev.txt
streamlit run app.py
```

Pruebas:

```bash
pytest              # suite principal (~1 min)
pytest -m slow      # además, ejecuta la app con cada una de las 94 industrias (~6 min)
```

## De dónde vienen los datos

Todos los datos son CSV versionados en `data/`. No hay archivos Excel y la app no descarga nada en tiempo de ejecución. Cada fila lleva `source`, `url`, `as_of` (fecha de los datos), `retrieved_at` (fecha de extracción) y `license`. Una prueba falla si alguna fila no tiene procedencia.

| Fuente | Qué aporta | Fecha de los datos | Condiciones de uso |
|---|---|---|---|
| [Aswath Damodaran](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datacurrent.html) (EE. UU.) | 62 métricas para 94 industrias (betas, beta total, costo de capital, márgenes, múltiplos, crecimiento, reinversión, capex), 10 clases de capitalización y prima de riesgo implícita | 2026-01-09 | Uso libre ("no strings attached", [reglas de uso](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datahistory.html#rules)); solo agregados por industria |
| [Banco Central Europeo](https://data.ecb.europa.eu/) | Tipo de cambio EUR/USD de referencia y tipo a 10 años de la curva AAA de la zona euro (tasa libre de riesgo en EUR) | ver `data/fx_rates.csv` | Uso libre citando al BCE ([aviso legal](https://www.ecb.europa.eu/services/disclaimer/html/index.en.html)) |
| [SEC EDGAR](https://www.sec.gov/data-research/sec-markets-data) Form D | Ofertas privadas: importe ofrecido y vendido, inversores, grupo de industria y rango de ingresos (empresa a empresa) | jul-2025 a jun-2026 | Información pública de sec.gov, redistribuible sin permiso ([política](https://www.sec.gov/about/privacy-information)) |
| SEC EDGAR Form C | Crowdfunding: estados financieros de startups pequeñas (ingresos, beneficio neto, caja, deuda, empleados) y condiciones de la oferta | jul-2025 a jun-2026 | Igual que Form D |
| SEC EDGAR S-1 + XBRL | Empresas que presentaron un S-1 con ingresos en XBRL; sin SPACs, trusts ni fondos | oct-2025 a sep-2026 | Igual que Form D |
| Supuestos propios | Parámetros por etapa (semilla a madura) | n/d | Ilustrativos, editables en la app |

Los datos de la SEC son por empresa. **No se guardan personas, firmantes, direcciones ni teléfonos**, solo datos de la empresa, y cada fila enlaza a su presentación en EDGAR. Ni Form D ni Form C informan la valoración. Form C tampoco informa la industria.

`data/sources.csv` registra también las fuentes candidatas investigadas y su estado: FRED, Kenneth French, Pablo Fernandez, Kroll, Carta y PitchBook-NVCA.

| Archivo | Contenido |
|---|---|
| `data/industry_metrics.csv` | Métricas por industria en formato largo (esquema único) |
| `data/industry_crosswalk.csv` | Equivalencia de la clasificación de cada fuente a la taxonomía propia (`industry_std`) y sector |
| `data/metric_definitions.csv` | Etiqueta, unidad, periodicidad y **topes configurables** de cada métrica |
| `data/size_class_metrics.csv` | Riesgo y múltiplos por decil de capitalización (proxy de madurez) |
| `data/market_metrics.csv` | Series de mercado: bono del Tesoro, prima implícita, curva AAA |
| `data/fx_rates.csv` | Tipos de cambio con fecha |
| `data/stage_assumptions.csv` | Supuestos por etapa |
| `data/damodaran_manifest.csv` | Páginas extraídas, número de industrias y hash del HTML |
| `data/sec_form_d.csv.gz`, `sec_form_c.csv.gz`, `sec_s1.csv.gz` | Una fila por empresa con su última presentación (comprimidos) |
| `data/sec_form_d_funds.csv.gz` | Vehículos de VC (fondos y SPVs) que presentaron Form D |
| `data/sec_manifest.csv` | Archivos de la SEC usados en cada generación |

### Limpieza aplicada

- Los porcentajes en texto ("40.20%") se guardan como decimales (0,402), y "NA" pasa a NaN.
- Los nombres de industria se normalizan en `industry_std` y el original se conserva en `industry_original`, incluida la errata "Heathcare Information and Technology".
- Los valores extremos se recortan con los topes de `metric_definitions.csv` y la app avisa del recorte. Ejemplos: 205 % de crecimiento esperado en Air Transport y 1.414 % de reinversión en Software (Internet). El recorte se puede desactivar en la barra lateral.
- Las filas "Total Market" son referencias, no industrias seleccionables. Si una industria no tiene una métrica, se usa el total de mercado sin financieras y la app lo indica.

### Regenerar los datos

Los scripts documentan cómo se generó cada CSV. No hacen falta para ejecutar la app.

```bash
python scripts/build_damodaran.py --refresh   # vuelve a descargar las 12 páginas
python scripts/build_ecb.py
python scripts/build_sec.py --quarters 4      # requiere SEC_USER_AGENT (ver abajo)
pytest
```

La SEC exige que cada descarga se identifique con un nombre y un correo de contacto. Se pasa por variable de entorno y nunca se commitea:

```bash
set SEC_USER_AGENT=valorador-startups tu@correo.com
```

## Cómo añadir una fuente nueva

1. Crea `src/sources/<fuente>.py` con una función que devuelva un DataFrame con el esquema único y regístrala:

   ```python
   from src.sources import register

   @register("mi_fuente")
   def load():
       df = pd.read_csv(DATA / "industry_metrics.csv")
       return df[df["source"] == "mi_fuente"]
   ```

2. Escribe `scripts/build_<fuente>.py`. Debe generar las filas con procedencia completa y reemplazar solo las de su fuente en `data/industry_metrics.csv`.
3. Añade sus industrias a `data/industry_crosswalk.csv` y la fuente a `data/sources.csv`.
4. Ejecuta `pytest`. Las pruebas comprueban la procedencia, la cobertura de la tabla de equivalencias y que la fuente aparece en el registro.

`src/valuation.py` no se toca: solo recibe números ya resueltos. Hay una prueba que registra una fuente ficticia y verifica que la valoración funciona sin cambios.

Si una fuente es de pago o prohíbe redistribuir, sus datos no se commitean. El módulo debe devolver un DataFrame vacío cuando falten sus datos, y la app seguirá funcionando con las demás. Las claves de API van en variables de entorno.

## Supuestos propios

- **Parámetros por etapa** (`data/stage_assumptions.csv`): IRR objetivo, probabilidad de supervivencia, dilución futura, prima de iliquidez y descuento por iliquidez. Son ilustrativos y se pueden editar en la pestaña «Supuestos».
- **Tasa de impuestos marginal** del 25 % por defecto. Se usa la marginal y no la efectiva de la industria, porque es la que pagará la startup cuando gane dinero.
- **Tasa de descuento**: empieza en el costo de capital de la startup y converge en los años 6 a 10 a una tasa madura (beta de mercado y D/E de la industria).
- **EBITDA** = margen operativo + D&A/ventas de la industria (EBITDA/ventas − margen operativo).
- **Prima de riesgo** implícita de EE. UU. de Damodaran. En EUR se aplica la misma prima a la tasa libre de riesgo AAA de la zona euro (mercado maduro, sin prima país).
- **Escenarios y Monte Carlo**: factores, desviaciones y correlaciones por defecto elegidos por el autor. Todos son editables.

## Estructura

```
app.py                 interfaz Streamlit
src/sources/           un módulo por fuente con la interfaz común
src/data.py            carga de CSV, taxonomía y resolución de prioridades
src/valuation.py       beta, DCF, método VC, múltiplos, escenarios
src/montecarlo.py      simulación
src/fund.py            métricas de fondos: DPI, RVPI, TVPI, XIRR, curva J, proyección
src/comparables.py     empresas de la SEC: búsqueda, comparación y precarga
src/statements.py      lectura de estados financieros (Excel, CSV, PDF)
src/company_store.py   biblioteca de empresas (Firestore o archivo local)
src/charts.py          gráficos y formato de números
data/                  CSV con procedencia
scripts/               un script por fuente
tests/                 pruebas
```

## Despliegue en Google Cloud Run

La imagen (`Dockerfile`) solo incluye la app y los datos. Región por defecto: `europe-southwest1` (Madrid).

Despliegue manual, una sola vez:

```bash
gcloud config set project <ID_DEL_PROYECTO>
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy valorador-startups --source . --region europe-southwest1 --allow-unauthenticated --session-affinity --min-instances 0 --max-instances 2 --memory 1Gi
```

La biblioteca de empresas usa **Firestore** (base de datos nativa en `europe-southwest1`) cuando la app corre en Cloud Run. La cuenta de servicio del servicio necesita `roles/datastore.user`, y no hace falta ninguna clave. En local se guarda en `.local_companies/` (no se versiona). Para usar Firestore en local: `VALORADOR_STORE=firestore` y `gcloud auth application-default login`. De los archivos subidos solo se guardan las cifras extraídas, no el archivo.

Despliegue continuo: crea un repositorio de Artifact Registry llamado `valorador` en la región y un activador de Cloud Build conectado al repositorio de GitHub que use `cloudbuild.yaml`. Cada push a `main` ejecuta las pruebas, construye la imagen y la despliega.

## Hoja de ruta

- Benchmarks de rentabilidad de fondos (DPI/TVPI por cosecha) cuando haya una fuente verificada y redistribuible.
- FRED para la tasa libre de riesgo de EE. UU. al día.
- Datos regionales de Damodaran (Europa, global), convertidos una sola vez de Excel a CSV.

## Licencia

Código bajo licencia MIT. Los datos de terceros conservan sus condiciones de uso, documentadas en cada fila y en `data/sources.csv`.
