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
                    data = stackstac.stack(
                        item, assets=["B08", "B11", "B04"], bounds_latlon=bbox, 
                        epsg=32720, resolution=10, resampling=Resampling.cubic
                    ).squeeze()
                    
                    # Renombrar bandas a 8, 11, 4
                    data = data.assign_coords(band=["8", "11", "4"])
                    
                    fname = f"T{item.properties.get('s2:mgrs_tile')}_{item.datetime.strftime('%Y%m%d')}.tif"
                    
                    # CAMBIO CLAVE: Usar archivo temporal para descarga
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                        tmp_path = tmp_file.name
                        data.rio.to_raster(tmp_path)
                    
                    # Leer el archivo y ofrecer descarga
                    with open(tmp_path, 'rb') as f:
                        file_bytes = f.read()
                    
                    # Limpiar archivo temporal
                    os.unlink(tmp_path)
                    
                    status.update(label=f"¡Listo para descargar! {fname}", state="complete")
                    
                    # Botón de descarga
                    st.download_button(
                        label=f"📥 Descargar {fname}",
                        data=file_bytes,
                        file_name=fname,
                        mime="image/tiff"
                    )
                    
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption(f"9 de Julio, Argentina | Búsqueda inteligente +/- 10 imágenes | Deployado en Streamlit Cloud")
