import streamlit as st
import os
import pystac_client
import planetary_computer
import stackstac
import rioxarray
import folium
import numpy as np
from streamlit_folium import st_folium
from rasterio.enums import Resampling
from datetime import datetime, timedelta
from folium.plugins import Draw
import tempfile
from PIL import Image
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sentinel-2 HD Downloader", layout="wide", page_icon="🛰️")
st.title("🛰️ Sentinel-2 HD Downloader")

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("1. Filtros Temporales")
    
    # Selector de Mes y Año
    meses = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    col_m, col_a = st.columns(2)
    with col_m:
        mes_nombre = st.selectbox("Mes", meses, index=datetime.now().month - 1)
    with col_a:
        anio = st.number_input("Año", min_value=2015, max_value=datetime.now().year, value=datetime.now().year)
    
    mes_num = meses.index(mes_nombre) + 1
    fecha_referencia = datetime(anio, mes_num, 1)

    max_cloud = st.slider("Cobertura máxima de nubes (%)", 0, 100, 10)
    
    st.markdown("---")
    st.header("2. Configuración de Salida")
    
    st.write("📥 **Descarga directa al navegador**")
    st.write("✅ **Bandas:** 8 (NIR), 11 (SWIR), 4 (Red)")
    st.write("✅ **Resolución:** 10 metros")
    st.write("✅ **Remuestreo:** Cubic Convolution")
    
    st.markdown("---")
    st.header("3. Formato de descarga")
    formato_descarga = st.radio(
        "Selecciona formato:",
        ["GeoTIFF (para GIS)", "JPG (para visualizar)", "Ambos"],
        help="GeoTIFF: mantiene georreferenciación para ArcGIS/QGIS\nJPG: para abrir en cualquier visor de fotos"
    )

# --- MAPA INTERACTIVO ---
st.subheader("1. Selecciona el área de interés (AOI)")
m = folium.Map(location=[-35.444, -60.884], zoom_start=13)
Draw(export=False, draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':True}).add_to(m)
map_data = st_folium(m, width=1200, height=450, key="main_map")

bbox = None
if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    st.success(f"Área definida: {bbox}")
else:
    st.warning("⚠️ Dibuja un rectángulo sobre el mapa.")

# --- LÓGICA DE BÚSQUEDA ---
if bbox:
    st.subheader(f"2. Imágenes cerca de {mes_nombre} {anio}")
    
    if st.button("🔍 Buscar (10 antes y 10 después)"):
        with st.spinner("Consultando catálogo..."):
            try:
                catalog = pystac_client.Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                    modifier=planetary_computer.sign_inplace,
                )
                
                # Búsqueda 1: 10 imágenes ANTES de la fecha de referencia
                search_before = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=f"2015-06-01/{fecha_referencia.isoformat()}",
                    max_items=10,
                    query={"eo:cloud_cover": {"lt": max_cloud}},
                    sortby=[{"field": "properties.datetime", "direction": "desc"}]
                )
                
                # Búsqueda 2: 10 imágenes DESPUÉS de la fecha de referencia
                search_after = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=f"{fecha_referencia.isoformat()}/{datetime.now().isoformat()}",
                    max_items=10,
                    query={"eo:cloud_cover": {"lt": max_cloud}},
                    sortby=[{"field": "properties.datetime", "direction": "asc"}]
                )
                
                # Combinar y ordenar
                scenes = list(search_before.items()) + list(search_after.items())
                scenes.sort(key=lambda x: x.datetime)
                
                if scenes:
                    st.session_state['scenes_list'] = scenes
                else:
                    st.error("No se encontraron imágenes con esos filtros.")
            except Exception as e:
                st.error(f"Error: {e}")

    if 'scenes_list' in st.session_state:
        scenes = st.session_state['scenes_list']
        scene_options = {
            f"{s.datetime.strftime('%d/%m/%Y')} | Nubes: {s.properties['eo:cloud_cover']:.1f}% | Tile: {s.properties.get('s2:mgrs_tile')}": i 
            for i, s in enumerate(scenes)
        }
        selected_label = st.selectbox("Elegir imagen:", list(scene_options.keys()))
        item = scenes[scene_options[selected_label]]

        # Preview
        with st.expander("🖼️ Vista Previa Rápida", expanded=True):
            if st.button("Generar Preview"):
                try:
                    prev = stackstac.stack(item, assets=["B08", "B11", "B04"], bounds_latlon=bbox, epsg=32720, resolution=60).squeeze().compute()
                    img = np.moveaxis(prev.values, 0, -1)
                    st.image(np.clip(img / 3500, 0, 1), use_container_width=True)
                except Exception as e:
                    st.error(f"Error preview: {e}")

        if st.button("🚀 Descargar Imagen HD (10m)"):
            try:
                with st.status("Procesando descarga HD...", expanded=True) as status:
                    # Procesar datos
                    data = stackstac.stack(
                        item, assets=["B08", "B11", "B04"], bounds_latlon=bbox, 
                        epsg=32720, resolution=10, resampling=Resampling.cubic
                    ).squeeze()
                    
                    # Renombrar bandas a 8, 11, 4
                    data = data.assign_coords(band=["8", "11", "4"])
                    
                    fname_base = f"T{item.properties.get('s2:mgrs_tile')}_{item.datetime.strftime('%Y%m%d')}"
                    
                    # Convertir a numpy para procesar
                    img_array = data.values
                    img_array = np.moveaxis(img_array, 0, -1)  # (bands, h, w) -> (h, w, bands)
                    
                    status.update(label="Generando archivos...", state="running")
                    
                    # --- OPCIÓN 1: GeoTIFF ---
                    if formato_descarga in ["GeoTIFF (para GIS)", "Ambos"]:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_tif:
                            data.rio.to_raster(tmp_tif.name)
                            with open(tmp_tif.name, 'rb') as f:
                                tif_bytes = f.read()
                            os.unlink(tmp_tif.name)
                        
                        st.download_button(
                            label=f"📥 Descargar {fname_base}.tif (GeoTIFF)",
                            data=tif_bytes,
                            file_name=f"{fname_base}.tif",
                            mime="image/tiff",
                            key="download_tif"
                        )
                    
                    # --- OPCIÓN 2: JPG ---
                    if formato_descarga in ["JPG (para visualizar)", "Ambos"]:
                        # Normalizar valores para visualización (0-255)
                        img_normalized = np.clip(img_array / 3500, 0, 1)  # Ajuste de brillo
                        img_8bit = (img_normalized * 255).astype(np.uint8)
                        
                        # Crear imagen PIL y guardar como JPG
                        img_pil = Image.fromarray(img_8bit)
                        
                        # Guardar en buffer
                        jpg_buffer = io.BytesIO()
                        img_pil.save(jpg_buffer, format='JPEG', quality=95)
                        jpg_bytes = jpg_buffer.getvalue()
                        
                        st.download_button(
                            label=f"📷 Descargar {fname_base}.jpg (Visualización)",
                            data=jpg_bytes,
                            file_name=f"{fname_base}.jpg",
                            mime="image/jpeg",
                            key="download_jpg"
                        )
                    
                    status.update(label=f"¡Listo para descargar!", state="complete")
                    
                    st.info("💡 **Tip:** El JPG es ideal para vista rápida en Windows. El GeoTIFF mantiene toda la información geoespacial.")
                    
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("---")
st.caption(f"9 de Julio, Argentina | Búsqueda inteligente +/- 10 imágenes | Deployado en Streamlit Cloud")
