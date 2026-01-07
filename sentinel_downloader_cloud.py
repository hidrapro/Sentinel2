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
st.set_page_config(page_title="Satellite HD Downloader", layout="wide", page_icon="🛰️")
st.title("🛰️ Multi-Satellite HD Downloader")

# --- DICCIONARIO DE CONFIGURACIÓN POR SATÉLITE ---
SAT_CONFIG = {
    "Sentinel-2": {
        "collection": "sentinel-2-l2a",
        "assets": ["B08", "B11", "B04"],
        "res": 10,
        "tile_key": "s2:mgrs_tile",
        "cloud_key": "eo:cloud_cover",
        "scale": 1.0,
        "offset": 0.0
    },
    "Landsat 8/9": {
        "collection": "landsat-c2-l2",
        "assets": ["nir08", "swir16", "red"],
        "res": 30,
        "tile_key": "landsat:wrs_path",
        "cloud_key": "eo:cloud_cover",
        "scale": 0.0000275,
        "offset": -0.2
    }
}

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("1. Selección de Plataforma")
    sat_choice = st.selectbox("Satélite", list(SAT_CONFIG.keys()))
    conf = SAT_CONFIG[sat_choice]

    st.header("2. Filtros Temporales")
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    col_m, col_a = st.columns(2)
    with col_m:
        mes_nombre = st.selectbox("Mes", meses, index=datetime.now().month - 1)
    with col_a:
        anio = st.number_input("Año", min_value=2013 if "Landsat" in sat_choice else 2015, max_value=datetime.now().year, value=datetime.now().year)
    
    mes_num = meses.index(mes_nombre) + 1
    fecha_referencia = datetime(anio, mes_num, 1)
    max_cloud = st.slider("Cobertura máxima de nubes (%)", 0, 100, 10)
    
    st.markdown("---")
    st.header("3. Configuración de Salida")
    res_final = st.number_input("Resolución de descarga (m)", value=conf["res"], min_value=10)
    
    formato_descarga = st.radio("Formato:", ["GeoTIFF (GIS)", "JPG (Visual)", "GIF Animado", "Todos"])
    
    percentil_bajo, percentil_alto = 2, 98
    if "JPG" in formato_descarga or "GIF" in formato_descarga or "Todos" == formato_descarga:
        metodo_norm = st.radio("Normalización:", ["Auto", "Manual"])
        if metodo_norm == "Manual":
            percentil_bajo = st.slider("Corte inf %", 0, 10, 2)
            percentil_alto = st.slider("Corte sup %", 90, 100, 98)
    
    if "GIF" in formato_descarga or "Todos" == formato_descarga:
        gif_duration = st.slider("ms por frame", 200, 2000, 500)
        # Se aumentó el límite máximo de imágenes de 30 a 40
        gif_max_images = st.slider("Máx imágenes", 3, 40, 12)

# --- MAPA ---
st.subheader("1. Área de Interés (AOI)")
m = folium.Map(location=[-35.444, -60.884], zoom_start=13)
LocateControl(auto_start=True).add_to(m)
Draw(draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':True}).add_to(m)
map_data = st_folium(m, width=1200, height=400, key="main_map")

bbox = None
if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
else:
    st.warning("Dibuja un rectángulo para comenzar.")

# --- PROCESAMIENTO ---
def normalize_image_robust(img_array, percentile_low=2, percentile_high=98, scale=1.0, offset=0.0):
    """
    Normalización corregida:
    Calcula el estiramiento de contraste ANTES de recortar los valores.
    """
    img_work = np.copy(img_array).astype(np.float32)
    
    # Identificar datos válidos (DN > 0 y no NaN)
    valid_mask = (img_work > 0) & (~np.isnan(img_work))
    
    # Aplicar escala y offset (Landsat transforma DN a Reflectancia 0-1)
    # Sentinel mantiene sus valores originales (ej. 0-10000)
    img_work = img_work * scale + offset
    
    if np.sum(valid_mask) < 100:
        # Fallback básico si no hay datos suficientes
        return (np.clip(img_work, 0, 1) * 255).astype(np.uint8)
    
    # Calculamos percentiles sobre los valores reales escalados
    valid_pixels = img_work[valid_mask]
    p_low = np.percentile(valid_pixels, percentile_low)
    p_high = np.percentile(valid_pixels, percentile_high)
    
    # Estiramiento (Contrast Stretching)
    # Importante: No recortamos a [0,1] antes de este paso para no perder el histograma
    denom = max(1e-5, p_high - p_low)
    img_stretched = (img_work - p_low) / denom
    
    # Finalmente recortamos y convertimos a 8 bits (0-255)
    return (np.clip(img_stretched, 0, 1) * 255).astype(np.uint8)

def add_text_to_image(img_pil, text):
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    fs = max(12, min(w // 25, h // 12))
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except: font = ImageFont.load_default()
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    m = fs // 6
    x, y = (w - tw) // 2, h - th - max(4, h // 40) - m
    draw.rectangle([x-m, y-m, x+tw+m, y+th+m], fill=(0,0,0,160))
    draw.text((x, y), text, fill=(255,255,255), font=font)
    return img_pil

# --- LÓGICA DE BÚSQUEDA Y DESCARGA ---
if bbox:
    if st.button(f"🔍 Buscar en {sat_choice}"):
        with st.spinner("Conectando con Planetary Computer..."):
            try:
                catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
                common_args = {"collections": [conf["collection"]], "bbox": bbox, "query": {conf["cloud_key"]: {"lt": max_cloud}}}
                
                # Se aumentó max_items de 20 a 25 para permitir capturar hasta 50 imágenes totales
                s_before = catalog.search(**common_args, datetime=f"2013-01-01/{fecha_referencia.isoformat()}", max_items=25, sortby=[{"field":"properties.datetime","direction":"desc"}])
                s_after = catalog.search(**common_args, datetime=f"{fecha_referencia.isoformat()}/{datetime.now().isoformat()}", max_items=25, sortby=[{"field":"properties.datetime","direction":"asc"}])
                
                scenes = list(s_before.items()) + list(s_after.items())
                scenes.sort(key=lambda x: x.datetime)
                if scenes:
                    st.session_state['scenes_list'] = scenes
                    st.success(f"Encontradas {len(scenes)} imágenes.")
                else: st.error("Sin resultados.")
            except Exception as e: st.error(f"Error STAC: {e}")

    if 'scenes_list' in st.session_state:
        scenes = st.session_state['scenes_list']
        if "GIF" not in formato_descarga:
            scene_opts = {f"{s.datetime.strftime('%d/%m/%Y')} | Nubes: {s.properties[conf['cloud_key']]:.1f}%": i for i, s in enumerate(scenes)}
            idx = st.selectbox("Seleccionar imagen:", list(scene_opts.keys()))
            item = scenes[scene_opts[idx]]

            if st.button("🖼️ Vista Previa"):
                data = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=60).squeeze().compute()
                img = normalize_image_robust(
                    np.moveaxis(data.values, 0, -1), 
                    percentil_bajo, percentil_alto, 
                    conf["scale"], conf["offset"]
                )
                st.image(img, use_container_width=True)

            if st.button(f"🚀 Descargar {sat_choice} HD"):
                with st.status("Procesando..."):
                    data = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=res_final, resampling=Resampling.cubic).squeeze()
                    tile = item.properties.get(conf["tile_key"], "IMG")
                    fname = f"{sat_choice[0]}_{tile}_{item.datetime.strftime('%Y%m%d')}"
                    
                    if "GeoTIFF" in formato_descarga or "Todos" == formato_descarga:
                        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                            data.rio.to_raster(tmp.name)
                            with open(tmp.name, 'rb') as f: 
                                st.download_button(f"📥 {fname}.tif", f.read(), f"{fname}.tif")
                    
                    if "JPG" in formato_descarga or "Todos" == formato_descarga:
                        img_8bit = normalize_image_robust(
                            np.moveaxis(data.values, 0, -1), 
                            percentil_bajo, percentil_alto, 
                            conf["scale"], conf["offset"]
                        )
                        buf = io.BytesIO()
                        Image.fromarray(img_8bit).save(buf, format='JPEG', quality=95)
                        st.download_button(f"📷 {fname}.jpg", buf.getvalue(), f"{fname}.jpg")
        else:
            if st.button("🎬 Generar GIF Multi-Temporal"):
                with st.status("Procesando frames...") as status:
                    frames, skipped = [], 0
                    subset = scenes[:gif_max_images]
                    for i, s in enumerate(subset):
                        status.update(label=f"Frame {i+1}/{len(subset)}...")
                        data = stackstac.stack(s, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=conf["res"]*2).squeeze().compute()
                        img_raw = np.moveaxis(data.values, 0, -1)
                        
                        # Discriminador de bordes (usando valores brutos para detectar nodata)
                        if np.mean(np.any(np.isnan(img_raw) | (img_raw <= 0), axis=-1)) > 0.10:
                            skipped += 1; continue
                        
                        img_8bit = normalize_image_robust(
                            img_raw, percentil_bajo, percentil_alto, 
                            conf["scale"], conf["offset"]
                        )
                        frames.append(add_text_to_image(Image.fromarray(img_8bit), s.datetime.strftime('%d/%m/%Y')))
                    
                    if frames:
                        buf = io.BytesIO()
                        frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], duration=gif_duration, loop=0)
                        st.image(buf.getvalue())
                        st.download_button("🎬 Descargar GIF", buf.getvalue(), "serie_temporal.gif")
                        if skipped > 0: st.warning(f"Omitidas {skipped} imágenes incompletas.")
                    else: st.error("No hay suficientes imágenes válidas.")

st.markdown("---")
st.caption("Soporte Landsat 8/9 & Sentinel-2 | Corrección de contraste robusta | Normalización por satélite")