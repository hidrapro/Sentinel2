# 🛰️ Sentinel-2 HD Downloader

Aplicación web para descargar imágenes de alta resolución de Sentinel-2 desde Microsoft Planetary Computer.

## 🚀 Deployment en Streamlit Cloud

### Paso 1: Crear repositorio en GitHub

1. Crea un nuevo repositorio en GitHub (público o privado)
2. Sube estos archivos:
   - `sentinel_downloader_cloud.py`
   - `requirements.txt`
   - `README.md`

### Paso 2: Desplegar en Streamlit

1. Ve a [share.streamlit.io](https://share.streamlit.io)
2. Inicia sesión con tu cuenta de GitHub
3. Click en "New app"
4. Configura:
   - **Repository:** tu-usuario/nombre-repo
   - **Branch:** main
   - **Main file path:** sentinel_downloader_cloud.py
5. Click "Deploy"

¡Listo! Tu app estará disponible en: `https://tu-usuario-nombre-repo.streamlit.app`

## 📦 Características

- ✅ Búsqueda temporal inteligente (±10 imágenes alrededor de una fecha)
- ✅ Filtro de cobertura de nubes
- ✅ Vista previa rápida (60m)
- ✅ Descarga HD (10m, remuestreo cúbico)
- ✅ Bandas: NIR (B08), SWIR (B11), Red (B04)
- ✅ Descarga directa al navegador

## 🛠️ Uso local (opcional)

```bash
# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run sentinel_downloader_cloud.py
```

## 📝 Notas

- La app usa el catálogo de Microsoft Planetary Computer (gratuito)
- Las imágenes se procesan en memoria y se descargan directamente
- No se requiere configuración de credenciales

## 🌍 Área predeterminada

9 de Julio, Buenos Aires, Argentina (-35.444, -60.884)
