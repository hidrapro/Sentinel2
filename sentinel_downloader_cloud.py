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
# Nota Crítica: Landsat 1-3 usa la colección de Nivel 1 porque no existe Nivel 2 para MSS.
# Los alias de bandas en PC para MSS son diferentes: green, red, nir07, nir08.
SAT_CONFIG = {
    "Sentinel-2": {
        "collection": "sentinel-2-l2a",
        "platform": None,
        "assets": ["B08", "B11", "B04"],
        "res": 10,
        "tile_key": "s2:mgrs_tile",
        "cloud_key": "eo:cloud_cover",
        "scale": 1.0,
        "offset": 0.0,
        "min_year": 2015
    },
    "Landsat 8/9": {
        "collection": "landsat-c2-l2",
        "platform": ["landsat-8", "landsat-9"],
        "assets": ["nir08", "swir16", "red"],
        "res": 30,
        "tile_key": "landsat:wrs_path",
        "cloud_key": "eo:cloud_cover",
        "scale": 0.0000275,
        "offset": -0.2,
        "min_year": 2013
    },
    "Landsat 7": {
        "collection": "landsat-c2-l2",
        "platform": ["landsat-7"],
        "assets": ["nir08", "swir16", "red"],
        "res": 30,
        "tile_key": "landsat:wrs_path",
        "cloud_key": "eo:cloud_cover",
        "scale": 0.0000275,
        "offset": -0.2,
        "min_year": 1999
    },
    "Landsat 5": {
        "collection": "landsat-c2-l2",
        "platform": ["landsat-5"],
        "assets": ["nir08", "swir16", "red"], 
        "res": 30,
        "tile_key": "landsat:wrs_path",
        "cloud_key": "eo:cloud_cover",
        "scale": 0.0000275,
        "offset": -0.2,
        "min_year": 1984
    },
    "Landsat 4": {
        "collection": "landsat-c2-l2",
        "platform": ["landsat-4"],
        "assets": ["nir08", "swir16", "red"], 
        "res": 30,
        "tile_key": "landsat:wrs_path",
        "cloud_key": "eo:cloud_cover",
        "scale": 0.0000275,
        "offset": -0.2,
        "min_year": 1982
    },
    "Landsat 1-3 (MSS)": {
        "collection": "landsat-c2-l1",
        "platform": ["landsat-1", "landsat-2", "landsat-3"],
        "assets": ["nir08", "red", "green"], # Composición NIR-R-G (Falso color estándar MSS)
        "res": 60,
        "tile_key": "landsat:wrs_path",
        "cloud_key": "eo:cloud_cover",
        "scale": 1.0, # Nivel 1 no usa el escalado SR de Nivel 2
        "offset": 0.0,
        "min_year": 1972
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
        anio = st.number_input("Año", min_value=conf["min_year"], max_value=datetime.now().year, value=max(conf["min_year"], datetime.now().year))
    
    mes_num = meses.index(mes_nombre) + 1
    fecha_referencia = datetime(anio, mes_num, 1)
    max_cloud = st.slider("Cobertura máxima de nubes (Global %)", 0, 100, 15)
    
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
    
    # LISTA DE EXCLUSIÓN
    exclude_dates = []
    if 'scenes_before' in st.session_state and 'scenes_after' in st.session_state:
        st.markdown("---")
        st.header("5. Filtro Manual")
        all_candidates = st.session_state['scenes_before'] + st.session_state['scenes_after']
        all_candidates.sort(key=lambda x: x.datetime)
        all_dates = [s.datetime.strftime('%d/%m/%Y') for s in all_candidates]
        exclude_dates = st.multiselect("Fechas a ignorar:", options=all_dates)

    if "GIF" in formato_descarga or "Todos" == formato_descarga:
        st.subheader("Configuración GIF")
        gif_duration = st.slider("ms por frame", 200, 2000, 500)
        gif_max_images = st.slider("Cantidad de imágenes objetivo", 3, 50, 10)

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
    if st.button(f"🔍 Buscar Imágenes"):
        with st.spinner("Consultando catálogo STAC..."):
            try:
                catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
                
                query_params = {conf["cloud_key"]: {"lt": max_cloud}}
                
                if conf["platform"]:
                    query_params["platform"] = {"in": conf["platform"]}

                common_args = {
                    "collections": [conf["collection"]], 
                    "bbox": bbox, 
                    "query": query_params
                }
                
                fecha_inicio = fecha_referencia - timedelta(days=182)
                fecha_fin = fecha_referencia + timedelta(days=182)
                
                s_before = catalog.search(**common_args, datetime=f"{fecha_inicio.isoformat()}/{fecha_referencia.isoformat()}", max_items=100, sortby=[{"field":"properties.datetime","direction":"desc"}])
                s_after = catalog.search(**common_args, datetime=f"{fecha_referencia.isoformat()}/{fecha_fin.isoformat()}", max_items=100, sortby=[{"field":"properties.datetime","direction":"asc"}])
                
                st.session_state['scenes_before'] = list(s_before.items())
                st.session_state['scenes_after'] = list(s_after.items())
                
                st.success(f"Búsqueda finalizada. Candidatas encontradas: {len(st.session_state['scenes_before']) + len(st.session_state['scenes_after'])}")
                st.rerun()
            except Exception as e: st.error(f"Error STAC: {e}")

    if 'scenes_before' in st.session_state and 'scenes_after' in st.session_state:
        full_list = st.session_state['scenes_before'] + st.session_state['scenes_after']
        scenes_filtered = [s for s in full_list if s.datetime.strftime('%d/%m/%Y') not in exclude_dates]
        
        if "GIF" not in formato_descarga:
            scenes_filtered.sort(key=lambda x: x.datetime)
            scene_opts = {f"{s.datetime.strftime('%d/%m/%Y')} | Nubes: {s.properties[conf['cloud_key']]:.1f}%": i for i, s in enumerate(scenes_filtered)}
            if not scene_opts:
                st.warning("No hay imágenes disponibles en este rango.")
            else:
                idx = st.selectbox("Seleccionar imagen:", list(scene_opts.keys()))
                item = scenes_filtered[scene_opts[idx]]
                if st.button("🖼️ Vista Previa"):
                    data = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=conf["res"]*2).squeeze().compute()
                    img = normalize_image_robust(np.moveaxis(data.values, 0, -1), percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                    st.image(img, use_container_width=True)
                
                if st.button(f"🚀 Descargar HD"):
                    with st.status("Procesando bandas..."):
                        data = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=res_final, resampling=Resampling.cubic).squeeze()
                        tile = item.properties.get(conf["tile_key"], "IMG")
                        fname = f"{sat_choice.replace(' ', '_')}_{item.datetime.strftime('%Y%m%d')}"
                        
                        if "GeoTIFF" in formato_descarga or "Todos" == formato_descarga:
                            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                                data.rio.to_raster(tmp.name)
                                with open(tmp.name, 'rb') as f: st.download_button(f"📥 {fname}.tif", f.read(), f"{fname}.tif")
                        
                        if "JPG" in formato_descarga or "Todos" == formato_descarga:
                            img_8bit = normalize_image_robust(np.moveaxis(data.values.compute(), 0, -1), percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                            buf = io.BytesIO()
                            Image.fromarray(img_8bit).save(buf, format='JPEG', quality=95)
                            st.download_button(f"📷 {fname}.jpg", buf.getvalue(), f"{fname}.jpg")
        else:
            # LÓGICA DE GIF
            if st.button("🎬 Generar GIF Multi-Temporal"):
                with st.status("Generando frames...") as status:
                    pool_before = [s for s in st.session_state['scenes_before'] if s.datetime.strftime('%d/%m/%Y') not in exclude_dates]
                    pool_after = [s for s in st.session_state['scenes_after'] if s.datetime.strftime('%d/%m/%Y') not in exclude_dates]
                    
                    target_total = gif_max_images
                    frames_list = []
                    
                    def process_pool(pool, limit, desc):
                        count = 0
                        for s in pool:
                            if count >= limit: break
                            current_date_str = s.datetime.strftime('%d/%m/%Y')
                            status.update(label=f"Procesando {desc}: {current_date_str}...")
                            try:
                                data = stackstac.stack(s, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=conf["res"]*2).squeeze().compute()
                                img_raw = np.moveaxis(data.values, 0, -1)
                                if np.mean(np.any(np.isnan(img_raw) | (img_raw <= 0), axis=-1)) > 0.20: continue
                                img_8bit = normalize_image_robust(img_raw, percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                                frames_list.append((s.datetime, add_text_to_image(Image.fromarray(img_8bit), current_date_str)))
                                count += 1
                            except: continue
                        return count

                    process_pool(pool_after, target_total // 2, "futuro")
                    process_pool(pool_before, target_total - len(frames_list), "pasado")

                    if frames_list:
                        frames_list.sort(key=lambda x: x[0])
                        just_images = [f[1] for f in frames_list]
                        buf = io.BytesIO()
                        just_images[0].save(buf, format='GIF', save_all=True, append_images=just_images[1:], duration=gif_duration, loop=0)
                        st.image(buf.getvalue())
                        st.download_button("🎬 Descargar GIF", buf.getvalue(), "serie_temporal.gif")
                    else:
                        st.error("No se encontraron imágenes aptas para el GIF.")

st.markdown("---")
st.caption("Notas: Landsat 1-3 (1972-1983, Sensor MSS 60m, Nivel 1), Landsat 4-5 (TM 30m), Landsat 7-9 (ETM+/OLI 30m).")