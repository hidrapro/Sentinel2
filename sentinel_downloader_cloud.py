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
        "platform": None,
        "assets": ["B08", "B11", "B04"],
        "res": 10,
        "tile_key": "s2:mgrs_tile",
        "cloud_key": "eo:cloud_cover",
        "scale": 1.0,
        "offset": 0.0,
        "min_year": 2015,
        "max_year": datetime.now().year
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
        "min_year": 2013,
        "max_year": datetime.now().year
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
        "min_year": 1999,
        "max_year": datetime.now().year
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
        "min_year": 1984,
        "max_year": 2013
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
        "min_year": 1982,
        "max_year": 1993
    },
    "Landsat 1-3 (MSS)": {
        "collection": "landsat-c2-l1",
        "platform": ["landsat-1", "landsat-2", "landsat-3"],
        "assets": ["nir08", "red", "green"], 
        "res": 60,
        "tile_key": "landsat:wrs_path",
        "cloud_key": "eo:cloud_cover",
        "scale": 1.0, 
        "offset": 0.0,
        "min_year": 1972,
        "max_year": 1983
    }
}

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("1. Selección de Plataforma")
    
    def sat_label_formatter(key):
        c = SAT_CONFIG[key]
        end = "Presente" if c["max_year"] == datetime.now().year else str(c["max_year"])
        return f"{key} ({c['min_year']} - {end})"

    sat_choice = st.selectbox(
        "Satélite", 
        options=list(SAT_CONFIG.keys()), 
        format_func=sat_label_formatter
    )
    conf = SAT_CONFIG[sat_choice]

    st.header("2. Filtros Temporales")
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    col_m, col_a = st.columns(2)
    with col_m:
        mes_nombre = st.selectbox("Mes", meses, index=datetime.now().month - 1)
    with col_a:
        anio = st.number_input(
            "Año", 
            min_value=conf["min_year"], 
            max_value=conf["max_year"], 
            value=conf["max_year"]
        )
    
    mes_num = meses.index(mes_nombre) + 1
    fecha_referencia = datetime(anio, mes_num, 1)
    max_cloud = st.slider("Cobertura máxima de nubes (Global %)", 0, 100, 15)
    
    st.markdown("---")
    st.header("3. Mapa Base")
    map_style = st.selectbox("Estilo del Mapa", ["OpenStreetMap", "Satélite (Esri)", "Topográfico (OpenTopo)"])
    
    st.header("4. Configuración de Salida")
    res_final = st.number_input("Resolución de descarga (m)", value=conf["res"], min_value=10)
    formato_descarga = st.radio("Formato:", ["GeoTIFF (GIS)", "JPG (Visual)", "GIF Animado", "Todos"])
    
    percentil_bajo, percentil_alto = 2, 98

# --- PROCESAMIENTO ---
def normalize_image_robust(img_array, percentile_low=2, percentile_high=98, scale=1.0, offset=0.0):
    """
    Normaliza la imagen estirando el contraste por cada canal de forma independiente.
    Corregido para manejar offsets negativos de Landsat Collection 2.
    """
    img_work = np.copy(img_array).astype(np.float32)
    img_work = img_work * scale + offset
    
    # Umbral de 'no-data' ajustado al offset
    # Si offset es -0.2, el valor 0 original ahora es -0.2. Usamos -0.199 como margen.
    nodata_threshold = offset + 1e-6
    
    if len(img_work.shape) == 3:
        normalized_channels = []
        for i in range(img_work.shape[2]):
            channel = img_work[:, :, i]
            # Identificar datos válidos: mayores al umbral de fondo y no NaNs
            valid_mask = (channel > nodata_threshold) & (~np.isnan(channel))
            
            if np.sum(valid_mask) < 100:
                normalized_channels.append(np.zeros_like(channel, dtype=np.uint8))
                continue
            
            p_low = np.percentile(channel[valid_mask], percentile_low)
            p_high = np.percentile(channel[valid_mask], percentile_high)
            
            denom = max(1e-5, p_high - p_low)
            stretched = (channel - p_low) / denom
            normalized_channels.append((np.clip(stretched, 0, 1) * 255).astype(np.uint8))
        
        return np.stack(normalized_channels, axis=2)
    else:
        valid_mask = (img_work > nodata_threshold) & (~np.isnan(img_work))
        if not np.any(valid_mask):
             return np.zeros_like(img_work, dtype=np.uint8)
        p_low = np.percentile(img_work[valid_mask], percentile_low)
        p_high = np.percentile(img_work[valid_mask], percentile_high)
        denom = max(1e-5, p_high - p_low)
        return (np.clip((img_work - p_low) / denom, 0, 1) * 255).astype(np.uint8)

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

# --- MAPA ---
st.subheader("1. Área de Interés (AOI)")
tile_urls = {"OpenStreetMap": "OpenStreetMap", "Satélite (Esri)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Topográfico (OpenTopo)": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"}
m = folium.Map(location=[-35.444, -60.884], zoom_start=13, tiles=tile_urls[map_style] if map_style == "OpenStreetMap" else tile_urls[map_style], attr="Tiles &copy; Esri / OpenTopoMap" if map_style != "OpenStreetMap" else None)
LocateControl(auto_start=True).add_to(m)
Draw(draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':True}).add_to(m)
map_data = st_folium(m, width=1200, height=400, key="main_map")

bbox = None
if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]

# --- LÓGICA DE BÚSQUEDA ---
if bbox:
    if st.button(f"🔍 Buscar Imágenes"):
        with st.spinner("Consultando catálogo STAC..."):
            catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
            query_params = {conf["cloud_key"]: {"lt": max_cloud}}
            if conf["platform"]: query_params["platform"] = {"in": conf["platform"]}
            
            common_args = {"collections": [conf["collection"]], "bbox": bbox, "query": query_params}
            fecha_inicio, fecha_fin = fecha_referencia - timedelta(days=182), fecha_referencia + timedelta(days=182)
            
            s_before = catalog.search(**common_args, datetime=f"{fecha_inicio.isoformat()}/{fecha_referencia.isoformat()}", max_items=50, sortby=[{"field":"properties.datetime","direction":"desc"}])
            s_after = catalog.search(**common_args, datetime=f"{fecha_referencia.isoformat()}/{fecha_fin.isoformat()}", max_items=50, sortby=[{"field":"properties.datetime","direction":"asc"}])
            
            st.session_state['scenes_before'] = list(s_before.items())
            st.session_state['scenes_after'] = list(s_after.items())
            st.rerun()

    if 'scenes_before' in st.session_state:
        all_scenes = st.session_state['scenes_before'] + st.session_state['scenes_after']
        all_scenes.sort(key=lambda x: x.datetime)
        
        if not all_scenes:
            st.warning("No se encontraron escenas.")
        else:
            scene_opts = {f"{s.datetime.strftime('%d/%m/%Y')} | Nubes: {s.properties[conf['cloud_key']]:.1f}%": i for i, s in enumerate(all_scenes)}
            idx_name = st.selectbox("Seleccionar imagen:", list(scene_opts.keys()))
            item = all_scenes[scene_opts[idx_name]]

            if st.button("🖼️ Vista Previa"):
                # Preview a resolución reducida para rapidez
                data = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=conf["res"]*2).squeeze().compute()
                img = normalize_image_robust(np.moveaxis(data.values, 0, -1), percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                st.image(img, use_container_width=True, caption=f"Previa balanceada (Canal NIR mapeado a ROJO)")

            if st.button("🚀 Descargar HD"):
                with st.status("Procesando..."):
                    data = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=res_final).squeeze()
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

st.markdown("---")
st.caption("Nota técnica: Landsat 1-3 usa el sensor MSS que carece de banda azul. La composición NIR-R-G es un Falso Color Vegetal. El balance de blancos se ha corregido normalizando canales por separado.")
st.caption("Ing. Luis A. Carnaghi (lcarnaghi@gmail.com) es el creador de la página.")