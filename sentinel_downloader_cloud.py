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
import imageio
import time

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

# --- FUNCIONES DE SOPORTE GEOGRÁFICO ---
def get_utm_epsg(bbox):
    """Calcula automáticamente el EPSG de la zona UTM correspondiente al área."""
    if not bbox: return 3857 # Fallback a Web Mercator si no hay área
    lon = (bbox[0] + bbox[2]) / 2
    lat = (bbox[1] + bbox[3]) / 2
    utm_zone = int((lon + 180) / 6) + 1
    epsg_base = 32600 if lat >= 0 else 32700
    return epsg_base + utm_zone

# --- ESTADO INICIAL Y PERSISTENCIA ---
if 'map_center' not in st.session_state:
    st.session_state['map_center'] = [-35.444, -60.884]
if 'map_zoom' not in st.session_state:
    st.session_state['map_zoom'] = 13
if 'bbox' not in st.session_state:
    st.session_state['bbox'] = None
if 'preview_img' not in st.session_state:
    st.session_state['preview_img'] = None
if 'preview_caption' not in st.session_state:
    st.session_state['preview_caption'] = ""
if 'anim_gif' not in st.session_state:
    st.session_state['anim_gif'] = None
if 'anim_vid' not in st.session_state:
    st.session_state['anim_vid'] = None

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
    formato_descarga = st.radio("Formato:", ["GeoTIFF (GIS)", "JPG (Visual)", "Animación (GIF/Video)", "Todos"])
    
    percentil_bajo, percentil_alto = 2, 98

    # --- FILTRO MANUAL DE FECHAS ---
    exclude_dates = []
    if 'scenes_before' in st.session_state and 'scenes_after' in st.session_state:
        st.markdown("---")
        st.header("5. Filtro Manual")
        all_candidates = st.session_state['scenes_before'] + st.session_state['scenes_after']
        all_candidates.sort(key=lambda x: x.datetime)
        all_dates = sorted(list(set([s.datetime.strftime('%d/%m/%Y') for s in all_candidates])))
        exclude_dates = st.multiselect("Ignorar estas fechas:", options=all_dates)

    # --- CONTROLES ESPECÍFICOS DE ANIMACIÓN ---
    if "Animación" in formato_descarga or "Todos" == formato_descarga:
        st.markdown("---")
        st.subheader("🎞️ Ajustes de Animación")
        anim_format = st.selectbox("Formato de salida", ["GIF Animado", "Video MP4", "Ambos"], index=1)
        gif_duration = st.slider("ms por frame (solo GIF)", 100, 2000, 500)
        video_fps = st.slider("Frames por segundo (solo Video)", 1, 5, 2)
        gif_max_images = st.slider("Cantidad de imágenes objetivo", 3, 50, 15)

# --- PROCESAMIENTO ---
def normalize_image_robust(img_array, percentile_low=2, percentile_high=98, scale=1.0, offset=0.0):
    img_work = np.copy(img_array).astype(np.float32)
    img_work = img_work * scale + offset
    nodata_threshold = offset + 1e-6
    
    if len(img_work.shape) == 3:
        normalized_channels = []
        for i in range(img_work.shape[2]):
            channel = img_work[:, :, i]
            valid_mask = (channel > nodata_threshold) & (~np.isnan(channel))
            if np.sum(valid_mask) < 50:
                normalized_channels.append(np.zeros_like(channel, dtype=np.uint8))
                continue
            
            p_low, p_high = np.percentile(channel[valid_mask], [percentile_low, percentile_high])
            denom = max(1e-5, p_high - p_low)
            stretched = (channel - p_low) / denom
            normalized_channels.append((np.clip(stretched, 0, 1) * 255).astype(np.uint8))
        return np.stack(normalized_channels, axis=2)
    else:
        valid_mask = (img_work > nodata_threshold) & (~np.isnan(img_work))
        if not np.any(valid_mask): return np.zeros_like(img_work, dtype=np.uint8)
        p_low, p_high = np.percentile(img_work[valid_mask], [percentile_low, percentile_high])
        denom = max(1e-5, p_high - p_low)
        return (np.clip((img_work - p_low) / denom, 0, 1) * 255).astype(np.uint8)

def add_text_to_image(img_pil, text):
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    fs = int(max(14, w * 0.04)) 
    try: 
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except: 
        font = ImageFont.load_default()
    
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    m = fs // 3 
    x, y = (w - tw) // 2, h - th - m * 2
    
    draw.rectangle([x-m, y-m, x+tw+m, y+th+m], fill=(0,0,0,160))
    draw.text((x, y), text, fill=(255,255,255), font=font)
    return img_pil

# --- MAPA ---
st.subheader("1. Área de Interés (AOI)")
tile_urls = {
    "OpenStreetMap": "OpenStreetMap", 
    "Satélite (Esri)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", 
    "Topográfico (OpenTopo)": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
}

m = folium.Map(
    location=st.session_state['map_center'], 
    zoom_start=st.session_state['map_zoom'], 
    tiles=tile_urls[map_style] if map_style == "OpenStreetMap" else tile_urls[map_style], 
    attr="Tiles &copy; Esri / OpenTopoMap" if map_style != "OpenStreetMap" else None
)

if st.session_state['bbox']:
    b = st.session_state['bbox']
    folium.Rectangle(
        bounds=[[b[1], b[0]], [b[3], b[2]]],
        color="#ff7800",
        fill=True,
        fill_opacity=0.2,
        weight=2,
        tooltip="Área Seleccionada"
    ).add_to(m)

LocateControl(auto_start=False).add_to(m)
Draw(draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':True}).add_to(m)

map_data = st_folium(
    m, 
    width=1200, 
    height=400, 
    key="main_map"
)

if map_data:
    if map_data.get('center'):
        new_lat = round(map_data['center']['lat'], 4)
        new_lng = round(map_data['center']['lng'], 4)
        if abs(new_lat - st.session_state['map_center'][0]) > 0.001 or \
           abs(new_lng - st.session_state['map_center'][1]) > 0.001:
            st.session_state['map_center'] = [new_lat, new_lng]
    
    if map_data.get('zoom'):
        if map_data['zoom'] != st.session_state['map_zoom']:
            st.session_state['map_zoom'] = map_data['zoom']
    
    if map_data.get('all_drawings') and len(map_data['all_drawings']) > 0:
        drawing = map_data['all_drawings'][-1]
        if drawing.get('geometry'):
            coords = drawing['geometry']['coordinates'][0]
            lons, lats = [c[0] for c in coords], [c[1] for c in coords]
            new_bbox = [min(lons), min(lats), max(lons), max(lats)]
            if st.session_state['bbox'] != new_bbox:
                st.session_state['bbox'] = new_bbox

# --- LÓGICA DE BÚSQUEDA ---
if st.session_state['bbox']:
    if st.button(f"🔍 Buscar Imágenes"):
        # Verificación de AOI válido
        bbox = st.session_state['bbox']
        if abs(bbox[2] - bbox[0]) < 1e-6 or abs(bbox[3] - bbox[1]) < 1e-6:
            st.error("El área seleccionada es inválida o demasiado pequeña.")
        else:
            with st.spinner("Consultando catálogo STAC..."):
                try:
                    catalog = pystac_client.Client.open(
                        "https://planetarycomputer.microsoft.com/api/stac/v1", 
                        modifier=planetary_computer.sign_inplace
                    )
                    query_params = {conf["cloud_key"]: {"lt": max_cloud}}
                    if conf["platform"]: query_params["platform"] = {"in": conf["platform"]}
                    
                    fecha_inicio, fecha_fin = fecha_referencia - timedelta(days=365), fecha_referencia + timedelta(days=365)
                    
                    # --- SISTEMA DE REINTENTOS PARA EVITAR APIERROR POR LATENCIA ---
                    max_retries = 3
                    items_found = []
                    for attempt in range(max_retries):
                        try:
                            search = catalog.search(
                                collections=[conf["collection"]],
                                bbox=bbox,
                                datetime=f"{fecha_inicio.isoformat()}/{fecha_fin.isoformat()}",
                                query=query_params,
                                max_items=100
                            )
                            items_found = list(search.items())
                            break # Éxito
                        except Exception as e:
                            if attempt < max_retries - 1:
                                time.sleep(2 ** attempt) # Espera exponencial
                                continue
                            else:
                                raise e # Re-lanzar error si fallan todos los intentos

                    if items_found:
                        st.session_state['scenes_before'] = [i for i in items_found if i.datetime < fecha_referencia.replace(tzinfo=i.datetime.tzinfo)]
                        st.session_state['scenes_after'] = [i for i in items_found if i.datetime >= fecha_referencia.replace(tzinfo=i.datetime.tzinfo)]
                        st.session_state['preview_img'] = None
                        st.session_state['anim_gif'] = None
                        st.session_state['anim_vid'] = None
                        st.rerun()
                    else:
                        st.error("No se encontraron imágenes en el área.")
                
                except pystac_client.exceptions.APIError as api_err:
                    st.error("Error de conexión con Planetary Computer. Esto suele ser un problema temporal de red o saturación del servidor. Por favor, intenta de nuevo en unos segundos.")
                    st.info(f"Detalle técnico: {str(api_err)}")
                except Exception as e:
                    st.error(f"Error inesperado: {str(e)}")

# --- MOSTRAR RESULTADOS ---
if 'scenes_before' in st.session_state:
    full_pool = st.session_state['scenes_before'] + st.session_state['scenes_after']
    all_scenes = [s for s in full_pool if s.datetime.strftime('%d/%m/%Y') not in exclude_dates]
    all_scenes.sort(key=lambda x: x.datetime)
    
    if not all_scenes:
        st.warning("No hay escenas disponibles (revisa el filtro de exclusión).")
    else:
        # Calcular EPSG dinámico para esta zona del mundo
        dynamic_epsg = get_utm_epsg(st.session_state['bbox'])

        if "Animación" not in formato_descarga and "Todos" != formato_descarga:
            scene_opts = {f"{s.datetime.strftime('%d/%m/%Y')} | Nubes: {s.properties[conf['cloud_key']]:.1f}%": i for i, s in enumerate(all_scenes)}
            idx_name = st.selectbox("Seleccionar imagen específica:", list(scene_opts.keys()))
            item = all_scenes[scene_opts[idx_name]]

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🖼️ Vista Previa"):
                    data_prev = stackstac.stack(item, assets=conf["assets"], bounds_latlon=st.session_state['bbox'], epsg=dynamic_epsg, resolution=conf["res"]*2).squeeze().compute()
                    img = normalize_image_robust(np.moveaxis(data_prev.values, 0, -1), percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                    st.session_state['preview_img'] = img
                    st.session_state['preview_caption'] = f"Previa: {idx_name}"
                
                if st.session_state['preview_img'] is not None:
                    st.image(st.session_state['preview_img'], use_container_width=True, caption=st.session_state['preview_caption'])

            with col2:
                if st.button("🚀 Descargar HD"):
                    with st.status("Procesando datos HD..."):
                        data = stackstac.stack(item, assets=conf["assets"], bounds_latlon=st.session_state['bbox'], epsg=dynamic_epsg, resolution=res_final).squeeze()
                        fname = f"{sat_choice.replace(' ', '_')}_{item.datetime.strftime('%Y%m%d')}"
                        if "GeoTIFF" in formato_descarga or "Todos" == formato_descarga:
                            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                                data.rio.to_raster(tmp.name)
                                with open(tmp.name, 'rb') as f: st.download_button(f"📥 {fname}.tif", f.read(), f"{fname}.tif")
                        if "JPG" in formato_descarga or "Todos" == formato_descarga:
                            data_np = data.compute().values
                            img_8bit = normalize_image_robust(np.moveaxis(data_np, 0, -1), percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                            buf = io.BytesIO()
                            Image.fromarray(img_8bit).save(buf, format='JPEG', quality=95)
                            st.download_button(f"📷 {fname}.jpg", buf.getvalue(), f"{fname}.jpg")

        if "Animación" in formato_descarga or "Todos" == formato_descarga:
            st.markdown("---")
            if st.button("🎬 Generar Serie Temporal (Video/GIF)"):
                frames_list = []
                pool = sorted(all_scenes, key=lambda x: (abs((x.datetime.replace(tzinfo=None) - fecha_referencia).days), x.properties[conf['cloud_key']]))
                
                with st.status("Procesando frames...", state="running") as status:
                    processed_count = 0
                    for s in pool:
                        if processed_count >= gif_max_images: break
                        try:
                            date_str = s.datetime.strftime('%d/%m/%Y')
                            status.update(label=f"Procesando frame {processed_count + 1}: {date_str}...")
                            data_f = stackstac.stack(s, assets=conf["assets"], bounds_latlon=st.session_state['bbox'], epsg=dynamic_epsg, resolution=conf["res"]*2).squeeze().compute()
                            img_np = np.moveaxis(data_f.values, 0, -1)
                            
                            nodata_mask = np.any(np.isnan(img_np) | (img_np <= 0), axis=-1)
                            if np.mean(nodata_mask) > 0.25: continue
                            
                            img_8bit = normalize_image_robust(img_np, percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                            
                            img_pil = Image.fromarray(img_8bit)
                            base_w = 800 
                            curr_w, curr_h = img_pil.size
                            if curr_w != base_w:
                                new_h = int(curr_h * (base_w / curr_w))
                                img_pil = img_pil.resize((base_w, new_h), resample=Image.LANCZOS)
                            
                            frames_list.append((s.datetime, add_text_to_image(img_pil, date_str)))
                            processed_count += 1
                        except: continue
                    
                    if frames_list:
                        frames_list.sort(key=lambda x: x[0])
                        pil_images = [f[1] for f in frames_list]
                        numpy_frames = [np.array(img) for img in pil_images]

                        if anim_format in ["GIF Animado", "Ambos"]:
                            buf_gif = io.BytesIO()
                            pil_images[0].save(buf_gif, format='GIF', save_all=True, append_images=pil_images[1:], duration=gif_duration, loop=0)
                            st.session_state['anim_gif'] = buf_gif.getvalue()

                        if anim_format in ["Video MP4", "Ambos"]:
                            try:
                                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_vid:
                                    imageio.mimwrite(tmp_vid.name, numpy_frames, fps=video_fps, format='FFMPEG', codec='libx264', quality=8)
                                    with open(tmp_vid.name, 'rb') as f:
                                        st.session_state['anim_vid'] = f.read()
                            except Exception as e:
                                st.error(f"Error en video: {e}")
                        
                        status.update(label="✅ Serie temporal lista", state="complete")
                    else:
                        st.error("No se pudieron generar frames válidos.")

            if st.session_state['anim_gif'] is not None:
                st.image(st.session_state['anim_gif'], caption="Serie Temporal Generada (GIF)")
                st.download_button("📥 Descargar GIF Animado", st.session_state['anim_gif'], "serie_satelital.gif")

            if st.session_state['anim_vid'] is not None:
                st.video(st.session_state['anim_vid'])
                st.download_button("📥 Descargar Video MP4", st.session_state['anim_vid'], "serie_satelital.mp4")

st.markdown("---")
st.caption("Notas: Landsat 1-3 (60m MSS), Landsat 4-9 (30m TM/OLI).")
st.caption("Ing. Luis A. Carnaghi (lcarnaghi@gmail.com) es el creador de la página.")
