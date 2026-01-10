import streamlit as st
import os
import pystac_client
import planetary_computer
import stackstac
import rioxarray
import folium
import numpy as np
import pandas as pd
from streamlit_folium import st_folium
from rasterio.enums import Resampling
from datetime import datetime, timedelta
from folium.plugins import Draw, LocateControl
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io
import cv2

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Satellite HD Downloader", layout="wide", page_icon="🛰️")
st.title("🛰️ Multi-Satellite HD Downloader")

# --- DICCIONARIO DE CONFIGURACIÓN POR SATÉLITE ---
SAT_CONFIG = {
    "Sentinel-2": {
        "collection": "sentinel-2-l2a",
        "platform": None,
        "assets": ["B08", "B11", "B04", "B03"], # NIR, SWIR1, RED, GREEN
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
        "assets": ["nir08", "swir16", "red", "green"],
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
        "assets": ["nir08", "swir16", "red", "green"],
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
        "assets": ["nir08", "swir16", "red", "green"], 
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
        "assets": ["nir08", "swir16", "red", "green"], 
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

# --- FUNCIONES AUXILIARES ---
def get_utm_epsg(lon, lat):
    """Calcula el código EPSG de la zona UTM correcta según coordenadas."""
    # Calcular zona UTM
    utm_zone = int((lon + 180) / 6) + 1
    
    # Determinar hemisferio y código EPSG
    if lat >= 0:
        # Hemisferio Norte: 326XX
        epsg_code = 32600 + utm_zone
    else:
        # Hemisferio Sur: 327XX
        epsg_code = 32700 + utm_zone
    
    return epsg_code

def normalize_image_robust(img_arr, p_low=2, p_high=98, scale=1.0, offset=0.0):
    """Normalización robusta de imagen con percentiles."""
    img = img_arr * scale + offset
    
    if img.ndim == 3:
        out = np.zeros_like(img, dtype=np.uint8)
        for i in range(img.shape[2]):
            band = img[:, :, i]
            # Filtrar valores válidos (no NaN y mayores a un umbral bajo)
            valid = band[(~np.isnan(band)) & (band > -0.5)]
            if valid.size > 100:  # Asegurar suficientes píxeles válidos
                vmin, vmax = np.percentile(valid, [p_low, p_high])
                # Evitar división por cero
                if vmax > vmin:
                    band_norm = (band - vmin) / (vmax - vmin) * 255
                    out[:, :, i] = np.clip(band_norm, 0, 255).astype(np.uint8)
        return out
    else:
        # Filtrar valores válidos
        valid = img[(~np.isnan(img)) & (img > -0.5)]
        if valid.size > 100:
            vmin, vmax = np.percentile(valid, [p_low, p_high])
            if vmax > vmin:
                img_norm = np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
                return np.stack([img_norm]*3, axis=-1)
        return np.zeros((*img.shape, 3), dtype=np.uint8)

def add_text_to_image(img, text):
    """Añadir texto a una imagen PIL."""
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([(10, 10), (w + 20, h + 20)], fill=(0, 0, 0, 180))
    draw.text((15, 15), text, fill=(255, 255, 255), font=font)
    return img

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
        anio = st.number_input("Año", min_value=conf["min_year"], max_value=conf["max_year"], value=conf["max_year"])
    
    mes_num = meses.index(mes_nombre) + 1
    fecha_referencia = datetime(anio, mes_num, 1)
    max_cloud = st.slider("Nubosidad máx. (%)", 0, 100, 15)
    
    st.markdown("---")
    st.header("3. Mapa y Salida")
    map_style = st.selectbox("Estilo del Mapa", ["OpenStreetMap", "Satélite (Esri)", "Topográfico (OpenTopo)"])
    res_final = st.number_input("Resolución descarga (m)", value=conf["res"], min_value=10)
    formato_descarga = st.radio("Formato:", ["GeoTIFF (GIS)", "JPG (Visual)", "Video MP4", "Todos"])
    
    percentil_bajo, percentil_alto = 2, 98

    # --- FILTRO MANUAL DE FECHAS ---
    exclude_dates = []
    if 'scenes_before' in st.session_state and 'scenes_after' in st.session_state:
        st.markdown("---")
        st.header("4. Filtro Manual")
        all_candidates = st.session_state['scenes_before'] + st.session_state['scenes_after']
        all_dates = sorted(list(set([s.datetime.strftime('%d/%m/%Y') for s in all_candidates])))
        exclude_dates = st.multiselect("Ignorar estas fechas:", options=all_dates)

    if "Video" in formato_descarga or "Todos" == formato_descarga:
        st.markdown("---")
        st.header("5. Video Temporal")
        video_fps = st.slider("Frames por segundo (FPS)", 1, 5, 2)
        video_max_images = st.slider("Máximo de frames", 3, 30, 15)

# --- MAPA ---
st.subheader("1. Área de Interés (AOI)")
tile_urls = {"OpenStreetMap": "OpenStreetMap", "Satélite (Esri)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Topográfico (OpenTopo)": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"}
m = folium.Map(location=[-35.444, -60.884], zoom_start=13, tiles=tile_urls[map_style] if map_style == "OpenStreetMap" else tile_urls[map_style], attr="Tiles &copy; Esri / OpenTopoMap" if map_style != "OpenStreetMap" else None)
LocateControl(auto_start=True).add_to(m)
Draw(draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':True}).add_to(m)
map_data = st_folium(m, width=1200, height=400, key="main_map")

bbox = None
epsg_code = None
if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    # Calcular centro del bbox para determinar zona UTM
    center_lon = (min(lons) + max(lons)) / 2
    center_lat = (min(lats) + max(lats)) / 2
    epsg_code = get_utm_epsg(center_lon, center_lat)
    st.info(f"📍 Zona UTM detectada: EPSG:{epsg_code}")

# --- LÓGICA DE BÚSQUEDA ---
if bbox:
    if st.button(f"🔍 Buscar Imágenes"):
        with st.spinner("Consultando catálogo STAC..."):
            catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
            query_params = {conf["cloud_key"]: {"lt": max_cloud}}
            if conf["platform"]: query_params["platform"] = {"in": conf["platform"]}
            common_args = {"collections": [conf["collection"]], "bbox": bbox, "query": query_params}
            
            fecha_inicio, fecha_fin = fecha_referencia - timedelta(days=365), fecha_referencia + timedelta(days=365)
            search = catalog.search(**common_args, datetime=f"{fecha_inicio.isoformat()}/{fecha_fin.isoformat()}", max_items=100)
            
            all_items = list(search.items())
            if all_items:
                st.session_state['scenes_before'] = [i for i in all_items if i.datetime < fecha_referencia.replace(tzinfo=i.datetime.tzinfo)]
                st.session_state['scenes_after'] = [i for i in all_items if i.datetime >= fecha_referencia.replace(tzinfo=i.datetime.tzinfo)]
                st.rerun()
            else:
                st.error("No se encontraron imágenes en el área.")

    if 'scenes_before' in st.session_state:
        full_pool = st.session_state['scenes_before'] + st.session_state['scenes_after']
        all_scenes = [s for s in full_pool if s.datetime.strftime('%d/%m/%Y') not in exclude_dates]
        all_scenes.sort(key=lambda x: x.datetime)
        
        if not all_scenes:
            st.warning("No hay escenas disponibles.")
        else:
            if formato_descarga != "Video MP4":
                scene_opts = {f"{s.datetime.strftime('%d/%m/%Y')} | Nubes: {s.properties[conf['cloud_key']]:.1f}%": i for i, s in enumerate(all_scenes)}
                idx_name = st.selectbox("Seleccionar imagen específica:", list(scene_opts.keys()))
                item = all_scenes[scene_opts[idx_name]]

                with st.expander("ℹ️ Información Técnica de la Escena"):
                    st.write(f"**ID:** `{item.id}`")
                    st.write(f"**Plataforma:** {item.properties.get('platform', 'N/A')}")
                    st.write(f"**Elevación Solar:** {item.properties.get('view:sun_elevation', 'N/A')}°")
                    st.write(f"**Cobertura de Nubes:** {item.properties.get(conf['cloud_key'], 0):.2f}%")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🖼️ Vista Previa"):
                        data_raw = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=epsg_code, resolution=conf["res"]*2).squeeze().compute()
                        img_np = np.moveaxis(data_raw.sel(band=conf["assets"][:3]).values, 0, -1)
                        img = normalize_image_robust(img_np, percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                        st.image(img, use_container_width=True, caption=f"Composición RGB: {idx_name}")
                with col2:
                    if st.button("🚀 Descargar HD"):
                        with st.status("Procesando datos HD..."):
                            data_raw = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=epsg_code, resolution=res_final).squeeze()
                            data_final = data_raw.sel(band=conf["assets"][:3])
                            fname = f"{sat_choice.replace(' ', '_')}_{item.datetime.strftime('%Y%m%d')}_RGB"
                            
                            if "GeoTIFF" in formato_descarga or "Todos" == formato_descarga:
                                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                                    data_final.rio.to_raster(tmp.name)
                                    with open(tmp.name, 'rb') as f: 
                                        st.download_button(f"📥 {fname}.tif", f.read(), f"{fname}.tif")
                                        
                            if "JPG" in formato_descarga or "Todos" == formato_descarga:
                                data_np = data_final.compute().values
                                img_input = np.moveaxis(data_np, 0, -1)
                                img_8bit = normalize_image_robust(img_input, percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                                buf = io.BytesIO()
                                Image.fromarray(img_8bit).save(buf, format='JPEG', quality=95)
                                st.download_button(f"📷 {fname}.jpg", buf.getvalue(), f"{fname}.jpg")

            # --- LÓGICA DE VIDEO MP4 ---
            if "Video" in formato_descarga or "Todos" == formato_descarga:
                st.markdown("---")
                if st.button("🎬 Generar Video MP4"):
                    frames_list = []
                    pool = sorted(all_scenes, key=lambda x: (abs((x.datetime.replace(tzinfo=None) - fecha_referencia).days), x.properties[conf['cloud_key']]))
                    with st.status("Generando frames...") as status:
                        processed = 0
                        for s in pool:
                            if processed >= video_max_images: break
                            try:
                                date_str = s.datetime.strftime('%d/%m/%Y')
                                status.update(label=f"Analizando frame {processed + 1}: {date_str}...")
                                data_f = stackstac.stack(s, assets=conf["assets"], bounds_latlon=bbox, epsg=epsg_code, resolution=conf["res"]*2).squeeze().compute()
                                check_np = data_f.sel(band=conf["assets"][0]).values
                                if np.mean(np.isnan(check_np) | (check_np <= 0)) > 0.20: continue
                                img_np = np.moveaxis(data_f.sel(band=conf["assets"][:3]).values, 0, -1)
                                img_8bit = normalize_image_robust(img_np, percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                                frames_list.append((s.datetime, add_text_to_image(Image.fromarray(img_8bit), date_str)))
                                processed += 1
                            except: continue
                        
                        if frames_list:
                            status.update(label="Generando video MP4...")
                            frames_list.sort(key=lambda x: x[0])
                            images_only = [np.array(f[1]) for f in frames_list]
                            
                            # Crear video con opencv
                            height, width = images_only[0].shape[:2]
                            
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                video = cv2.VideoWriter(tmp.name, fourcc, video_fps, (width, height))
                                
                                for frame in images_only:
                                    # Convertir RGB a BGR para OpenCV
                                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    video.write(frame_bgr)
                                
                                video.release()
                                
                                with open(tmp.name, 'rb') as f:
                                    video_bytes = f.read()
                                    st.video(video_bytes)
                                    st.download_button("📥 Descargar Video MP4", video_bytes, "serie_temporal.mp4", mime="video/mp4")
                                    st.success(f"✅ Video generado: {len(images_only)} frames a {video_fps} FPS")

st.markdown("---")
st.caption("Ing. Luis A. Carnaghi (lcarnaghi@gmail.com) - Creador.")
