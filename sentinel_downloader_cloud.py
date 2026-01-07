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
from folium.plugins import Draw, LocateControl
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sentinel-2 HD Downloader", layout="wide", page_icon="🛰️")
st.title("🛰️ Sentinel-2 HD Downloader")

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("1. Filtros Temporales")
    
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
        ["GeoTIFF (para GIS)", "JPG (para visualizar)", "GIF Animado (serie temporal)", "Todos"]
    )
    
    percentil_bajo = 2
    percentil_alto = 98
    if formato_descarga in ["JPG (para visualizar)", "GIF Animado (serie temporal)", "Todos"]:
        st.markdown("---")
        st.subheader("Ajustes de normalización")
        metodo_norm = st.radio("Método de normalización:", ["Automático (recomendado)", "Manual"])
        if metodo_norm == "Manual":
            percentil_bajo = st.slider("Corte inferior (%)", 0, 10, 2)
            percentil_alto = st.slider("Corte superior (%)", 90, 100, 98)
    
    if formato_descarga in ["GIF Animado (serie temporal)", "Todos"]:
        st.markdown("---")
        st.subheader("Configuración GIF")
        gif_duration = st.slider("Duración por frame (ms)", 200, 2000, 500, 50)
        gif_max_images = st.slider("Máx. imágenes en GIF", 3, 15, 8)

# --- MAPA INTERACTIVO ---
st.subheader("1. Selecciona el área de interés (AOI)")
# Mantenemos 9 de Julio como fallback si la geolocalización falla
m = folium.Map(location=[-35.444, -60.884], zoom_start=13)

# Añadir botón de geolocalización (Locate Me)
# auto_start=True intentará pedir permiso y centrar apenas cargue
LocateControl(
    auto_start=True,
    strings={"title": "Mostrar mi ubicación", "popup": "Estás aquí"},
    locateOptions={'enableHighAccuracy': True}
).add_to(m)

Draw(
    export=False, 
    draw_options={
        'polyline':False, 'polygon':False, 'circle':False, 
        'marker':False, 'circlemarker':False, 'rectangle':True
    }
).add_to(m)

map_data = st_folium(m, width=1200, height=450, key="main_map")

bbox = None
if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    st.success(f"Área definida: {bbox}")
else:
    st.warning("⚠️ Dibuja un rectángulo sobre el mapa. Usa el botón de mira (izquierda) para ir a tu ubicación.")

# --- FUNCIONES DE PROCESAMIENTO ---
def normalize_image_robust(img_array, percentile_low=2, percentile_high=98):
    valid_mask = (img_array > 0) & (~np.isnan(img_array))
    if np.sum(valid_mask) < 100:
        return (np.clip(img_array / 3000, 0, 1) * 255).astype(np.uint8)
    valid_values = img_array[valid_mask]
    p_low = np.percentile(valid_values, percentile_low)
    p_high = np.percentile(valid_values, percentile_high)
    if p_high - p_low < 1:
        p_low, p_high = np.min(valid_values), np.max(valid_values)
    img_stretched = np.clip((img_array - p_low) / (p_high - p_low), 0, 1)
    return (img_stretched * 255).astype(np.uint8)

def add_text_to_image(img_pil, text, position='bottom'):
    draw = ImageDraw.Draw(img_pil)
    img_width, img_height = img_pil.size
    font_size = max(12, min(img_width // 25, img_height // 12))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    bbox_text = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]
    padding_edge, bg_margin = max(4, img_height // 40), max(2, font_size // 6)
    x = (img_width - text_width) // 2
    y = img_height - text_height - padding_edge - bg_margin if position == 'bottom' else padding_edge + bg_margin
    draw.rectangle([x - bg_margin, y - bg_margin, x + text_width + bg_margin, y + text_height + bg_margin], fill=(0, 0, 0, 160))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return img_pil

# --- LÓGICA DE BÚSQUEDA ---
if bbox:
    st.subheader(f"2. Imágenes cerca de {mes_nombre} {anio}")
    if st.button("🔍 Buscar imágenes"):
        with st.spinner("Consultando catálogo..."):
            try:
                catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
                search_before = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"2015-06-01/{fecha_referencia.isoformat()}", max_items=10, query={"eo:cloud_cover": {"lt": max_cloud}}, sortby=[{"field": "properties.datetime", "direction": "desc"}])
                search_after = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{fecha_referencia.isoformat()}/{datetime.now().isoformat()}", max_items=10, query={"eo:cloud_cover": {"lt": max_cloud}}, sortby=[{"field": "properties.datetime", "direction": "asc"}])
                scenes = list(search_before.items()) + list(search_after.items())
                scenes.sort(key=lambda x: x.datetime)
                if scenes:
                    st.session_state['scenes_list'] = scenes
                    st.success(f"✅ Encontradas {len(scenes)} imágenes")
                else:
                    st.error("No se encontraron imágenes.")
            except Exception as e:
                st.error(f"Error: {e}")

    if 'scenes_list' in st.session_state:
        scenes = st.session_state['scenes_list']
        if formato_descarga != "GIF Animado (serie temporal)":
            st.subheader("3. Descarga de imagen individual")
            scene_options = {f"{s.datetime.strftime('%d/%m/%Y')} | Nubes: {s.properties['eo:cloud_cover']:.1f}%": i for i, s in enumerate(scenes)}
            selected_label = st.selectbox("Elegir imagen:", list(scene_options.keys()))
            item = scenes[scene_options[selected_label]]

            if st.button("🖼️ Generar Preview"):
                try:
                    prev = stackstac.stack(item, assets=["B08", "B11", "B04"], bounds_latlon=bbox, epsg=32720, resolution=60).squeeze().compute()
                    img_preview = normalize_image_robust(np.moveaxis(prev.values, 0, -1), percentil_bajo, percentil_alto)
                    st.image(img_preview, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

            if st.button("🚀 Descargar Imagen HD (10m)"):
                try:
                    with st.status("Procesando HD...", expanded=True):
                        data = stackstac.stack(item, assets=["B08", "B11", "B04"], bounds_latlon=bbox, epsg=32720, resolution=10, resampling=Resampling.cubic).squeeze()
                        data = data.assign_coords(band=["8", "11", "4"])
                        if formato_descarga in ["GeoTIFF (para GIS)", "Todos"]:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp:
                                data.rio.to_raster(tmp.name)
                                with open(tmp.name, 'rb') as f:
                                    st.download_button("📥 Descargar GeoTIFF", f.read(), "sentinel.tif")
                        if formato_descarga in ["JPG (para visualizar)", "Todos"]:
                            img_8bit = normalize_image_robust(np.moveaxis(data.values, 0, -1), percentil_bajo, percentil_alto)
                            buf = io.BytesIO()
                            Image.fromarray(img_8bit).save(buf, format='JPEG', quality=95)
                            st.download_button("📷 Descargar JPG", buf.getvalue(), "sentinel.jpg")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.subheader("3. Generar GIF Animado")
            if st.button("🎬 Generar GIF"):
                try:
                    with st.status("Generando frames...", expanded=True) as status:
                        frames, skipped = [], 0
                        for idx, scene in enumerate(scenes[:gif_max_images]):
                            status.update(label=f"Procesando {idx+1}/{len(scenes[:gif_max_images])}...")
                            data = stackstac.stack(scene, assets=["B08", "B11", "B04"], bounds_latlon=bbox, epsg=32720, resolution=30, resampling=Resampling.cubic).squeeze().compute()
                            img_array = np.moveaxis(data.values, 0, -1)
                            if np.mean(np.all(img_array <= 0, axis=-1)) > 0.05:
                                skipped += 1
                                continue
                            img_8bit = normalize_image_robust(img_array, percentil_bajo, percentil_alto)
                            frames.append(add_text_to_image(Image.fromarray(img_8bit), scene.datetime.strftime('%d/%m/%Y')))
                        
                        if frames:
                            buf = io.BytesIO()
                            frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], duration=gif_duration, loop=0, optimize=True)
                            st.image(buf.getvalue())
                            st.download_button("🎬 Descargar GIF", buf.getvalue(), "serie_temporal.gif")
                            if skipped > 0: st.warning(f"Omitidas {skipped} imágenes de borde.")
                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("---")
st.caption(f"Sentinel Downloader | Geolocalización habilitada | Filtro de bordes activo")