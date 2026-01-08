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
    max_cloud = st.slider("Cobertura máxima de nubes (Global %)", 0, 100, 10)
    
    st.markdown("---")
    st.header("3. Mapa Base")
    map_style = st.selectbox(
        "Estilo del Mapa",
        ["OpenStreetMap", "Satélite (Esri)", "Topográfico (OpenTopo)"]
    )
    
    st.markdown("---")
    st.header("4. Configuración de Salida")
    res_final = st.number_input("Resolución de descarga (m)", value=conf["res"], min_value=10)
    
    formato_descarga = st.radio("Formato:", ["GeoTIFF (GIS)", "JPG (Visual)", "GIF Animado", "Todos"])
    
    percentil_bajo, percentil_alto = 2, 98
    
    if "GIF" in formato_descarga or "Todos" == formato_descarga:
        st.subheader("Configuración GIF")
        gif_duration = st.slider("ms por frame", 200, 2000, 500)
        gif_max_images = st.slider("Máx imágenes", 3, 50, 12)
        # NUEVO: Filtro de nubes local
        st.info("☁️ El filtro global usa el metadato del satélite. El filtro local analiza tus píxeles.")
        local_cloud_filter = st.slider("Filtro Nubes Local (Umbral Brillo)", 150, 255, 220, help="Si el brillo promedio del AOI supera esto, se descarta el frame. Menor valor = más estricto.")

# --- MAPA ---
st.subheader("1. Área de Interés (AOI)")

tile_urls = {
    "OpenStreetMap": "OpenStreetMap",
    "Satélite (Esri)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Topográfico (OpenTopo)": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
}

attr_dict = {
    "Satélite (Esri)": "Tiles &copy; Esri &mdash; Source: Esri",
    "Topográfico (OpenTopo)": "Map data &copy; OpenStreetMap | Style: OpenTopoMap"
}

m = folium.Map(
    location=[-35.444, -60.884], 
    zoom_start=13,
    tiles=tile_urls[map_style] if map_style == "OpenStreetMap" else tile_urls[map_style],
    attr=attr_dict.get(map_style, None)
)

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
    img_work = np.copy(img_array).astype(np.float32)
    valid_mask = (img_work > 0) & (~np.isnan(img_work))
    img_work = img_work * scale + offset
    
    if np.sum(valid_mask) < 100:
        return (np.clip(img_work, 0, 1) * 255).astype(np.uint8)
    
    valid_pixels = img_work[valid_mask]
    p_low = np.percentile(valid_pixels, percentile_low)
    p_high = np.percentile(valid_pixels, percentile_high)
    
    denom = max(1e-5, p_high - p_low)
    img_stretched = (img_work - p_low) / denom
    
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
                
                # Expandimos la búsqueda para tener de donde elegir
                s_before = catalog.search(**common_args, datetime=f"2013-01-01/{fecha_referencia.isoformat()}", max_items=40, sortby=[{"field":"properties.datetime","direction":"desc"}])
                s_after = catalog.search(**common_args, datetime=f"{fecha_referencia.isoformat()}/{datetime.now().isoformat()}", max_items=40, sortby=[{"field":"properties.datetime","direction":"asc"}])
                
                scenes = list(s_before.items()) + list(s_after.items())
                scenes.sort(key=lambda x: x.datetime)
                if scenes:
                    st.session_state['scenes_list'] = scenes
                    st.success(f"Encontradas {len(scenes)} imágenes en catálogo.")
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
                img = normalize_image_robust(np.moveaxis(data.values, 0, -1), percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                st.image(img, use_container_width=True)

            if st.button(f"🚀 Descargar {sat_choice} HD"):
                with st.status("Procesando..."):
                    data = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=res_final, resampling=Resampling.cubic).squeeze()
                    tile = item.properties.get(conf["tile_key"], "IMG")
                    fname = f"{sat_choice[0]}_{tile}_{item.datetime.strftime('%Y%m%d')}"
                    
                    if "GeoTIFF" in formato_descarga or "Todos" == formato_descarga:
                        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                            data.rio.to_raster(tmp.name)
                            with open(tmp.name, 'rb') as f: st.download_button(f"📥 {fname}.tif", f.read(), f"{fname}.tif")
                    
                    if "JPG" in formato_descarga or "Todos" == formato_descarga:
                        img_8bit = normalize_image_robust(np.moveaxis(data.values, 0, -1), percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                        buf = io.BytesIO()
                        Image.fromarray(img_8bit).save(buf, format='JPEG', quality=95)
                        st.download_button(f"📷 {fname}.jpg", buf.getvalue(), f"{fname}.jpg")
        else:
            if st.button("🎬 Generar GIF Multi-Temporal"):
                with st.status("Procesando frames con filtro local...") as status:
                    frames, skipped_nodata, skipped_clouds = [], 0, 0
                    
                    # Seleccionamos una muestra uniforme del total encontrado para cubrir más tiempo
                    step = max(1, len(scenes) // gif_max_images)
                    subset = scenes[::step][:gif_max_images]
                    
                    for i, s in enumerate(subset):
                        status.update(label=f"Procesando frame {i+1}/{len(subset)}: {s.datetime.strftime('%d/%m/%Y')}...")
                        
                        # Carga de datos
                        data = stackstac.stack(s, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=conf["res"]*2).squeeze().compute()
                        img_raw = np.moveaxis(data.values, 0, -1)
                        
                        # 1. Filtro de datos faltantes (bordes negros)
                        if np.mean(np.any(np.isnan(img_raw) | (img_raw <= 0), axis=-1)) > 0.15:
                            skipped_nodata += 1; continue
                        
                        # 2. Normalización básica para análisis de brillo
                        img_8bit = normalize_image_robust(img_raw, percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                        
                        # 3. FILTRO DE NUBES LOCAL (NUEVO)
                        # Calculamos el brillo promedio de los píxeles. Las nubes disparan este valor.
                        local_brightness = np.mean(img_8bit)
                        if local_brightness > local_cloud_filter:
                            skipped_clouds += 1; continue
                        
                        frames.append(add_text_to_image(Image.fromarray(img_8bit), s.datetime.strftime('%d/%m/%Y')))
                    
                    if frames:
                        buf = io.BytesIO()
                        frames[0].save(buf, format='GIF', save_all=True, append_images=frames[1:], duration=gif_duration, loop=0)
                        st.image(buf.getvalue())
                        st.download_button("🎬 Descargar GIF", buf.getvalue(), "serie_temporal_filtrada.gif")
                        
                        # Reporte de calidad
                        col1, col2 = st.columns(2)
                        if skipped_clouds > 0: col1.warning(f"☁️ {skipped_clouds} frames descartados por nubes locales.")
                        if skipped_nodata > 0: col2.info(f"🌑 {skipped_nodata} frames descartados por falta de datos.")
                    else: 
                        st.error("No hay suficientes imágenes que pasen los filtros de calidad. Intenta subir el 'Umbral de Brillo' o aumentar la 'Cobertura Máxima Global'.")

st.markdown("---")
st.caption("Filtro local de nubes activado | Muestreo temporal inteligente | Basado en Reflectancia Superficial")