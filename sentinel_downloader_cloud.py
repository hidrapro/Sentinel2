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
import imageio

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Satelites LandSat y Sentinel 2", layout="wide", page_icon="🛰️")

# --- CSS PARA COMPACTAR LA BARRA LATERAL ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] hr {
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Visualizador y descarga de recortes")

# --- INICIALIZACIÓN DE ESTADO PARA BOTONES Y PERSISTENCIA ---
if "is_generating_preview" not in st.session_state:
    st.session_state.is_generating_preview = False
if "preview_image" not in st.session_state:
    st.session_state.preview_image = None
if "current_scene_id" not in st.session_state:
    st.session_state.current_scene_id = None
if "scene_coverage" not in st.session_state:
    st.session_state.scene_coverage = {}

# --- DICCIONARIO DE CONFIGURACIÓN POR SATÉLITE ---
SAT_CONFIG = {
    "Sentinel-2": {
        "collection": "sentinel-2-l2a",
        "platform": None,
        "assets": ["B08", "B11", "B04", "B03"], # NIR, SWIR1, RED, GREEN
        "res": 10,
        "tile_key": "s2:mgrs_tile",
        "cloud_key": "eo:cloud_cover",
        "scale": 0.0001,  # Sentinel-2 L2A viene en valores 0-10000, convertir a 0-1
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
    utm_zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        epsg_code = 32600 + utm_zone
    else:
        epsg_code = 32700 + utm_zone
    return epsg_code

def calculate_data_coverage(data_array):
    """
    Calcula el porcentaje de píxeles con datos válidos en un array.
    Retorna el porcentaje de cobertura de datos (0-100).
    """
    if data_array is None or data_array.size == 0:
        return 0.0
    
    # Considera como datos válidos los que no son NaN y son mayores a 0
    valid_data = ~(np.isnan(data_array) | (data_array <= 0))
    coverage = np.mean(valid_data) * 100
    return coverage

def normalize_image_robust(img_arr, p_low=2, p_high=98, scale=1.0, offset=0.0):
    """Normalización robusta de imagen con percentiles."""
    img = img_arr * scale + offset
    if img.ndim == 3:
        out = np.zeros_like(img, dtype=np.uint8)
        for i in range(img.shape[2]):
            band = img[:, :, i]
            valid = band[(~np.isnan(band)) & (band > -0.5)]
            if valid.size > 100:
                vmin, vmax = np.percentile(valid, [p_low, p_high])
                if vmax > vmin:
                    band_norm = (band - vmin) / (vmax - vmin) * 255
                    out[:, :, i] = np.clip(band_norm, 0, 255).astype(np.uint8)
        return out
    else:
        valid = img[(~np.isnan(img)) & (img > -0.5)]
        if valid.size > 100:
            vmin, vmax = np.percentile(valid, [p_low, p_high])
            if vmax > vmin:
                img_norm = np.clip((img - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
                return np.stack([img_norm]*3, axis=-1)
        return np.zeros((*img.shape, 3), dtype=np.uint8)

def add_text_to_image(img, text):
    """Añadir texto a una imagen PIL con tamaño equilibrado (5% del ancho)."""
    draw = ImageDraw.Draw(img)
    font_size = int(img.width * 0.05)
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "arial.ttf"
    ]
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except: continue
    if font is None: font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    padding = int(img.width * 0.01)
    draw.rectangle([padding, padding, bbox[2] - bbox[0] + 3*padding, bbox[3] - bbox[1] + 3*padding], fill=(0, 0, 0, 180))
    draw.text((2*padding, 2*padding), text, fill=(255, 255, 255), font=font)
    return img

# --- BARRA LATERAL ---
with st.sidebar:
    st.header("Parámetros")
    sat_choice = st.selectbox("Satélite:", list(SAT_CONFIG.keys()))
    conf = SAT_CONFIG[sat_choice]
    
    fecha_referencia = st.date_input("Fecha de referencia:", datetime.now().date(), 
                                     min_value=datetime(conf["min_year"], 1, 1).date(), 
                                     max_value=datetime(conf["max_year"], 12, 31).date())
    fecha_referencia = datetime.combine(fecha_referencia, datetime.min.time())
    
    st.divider()
    max_cloud = st.slider("Máx. Nubes (%):", 0, 100, 20)
    percentil_alto = st.slider("Percentil alto normalización:", 90, 100, 98)
    res_final = st.number_input("Resolución descarga (m):", 10, 120, conf["res"], step=10)
    
    st.divider()
    map_style = st.selectbox("Estilo mapa:", ["OpenStreetMap", "Satélite (Esri)", "Topográfico (OpenTopo)"])
    formato_descarga = st.selectbox("Formato descarga:", ["GeoTIFF", "JPG", "Video MP4", "Todos"])
    
    # NUEVA OPCIÓN: Calcular cobertura automáticamente
    st.divider()
    auto_calculate_coverage = st.checkbox("Calcular % datos automáticamente", value=False, 
                                          help="Calcula el porcentaje de datos válidos para cada escena (puede tomar tiempo)")
    
    exclude_dates = []
    if 'scenes_before' in st.session_state and 'scenes_after' in st.session_state:
        with st.expander("🔍 Filtro Manual de Fechas"):
            all_candidates = st.session_state['scenes_before'] + st.session_state['scenes_after']
            all_dates = sorted(list(set([s.datetime.strftime('%d/%m/%Y') for s in all_candidates])))
            exclude_dates = st.multiselect("Ignorar estas fechas:", options=all_dates)
    else:
        exclude_dates = []

    if "Video" in formato_descarga or formato_descarga == "Todos":
        with st.expander("🎬 Configuración Video"):
            video_fps = st.slider("FPS", 1, 5, 2)
            video_max_images = st.slider("Máx. frames", 3, 30, 15)

# --- MAPA ---
st.subheader("1. Área de Interés (AOI)")
tile_urls = {"OpenStreetMap": "OpenStreetMap", "Satélite (Esri)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Topográfico (OpenTopo)": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"}
m = folium.Map(location=[-35.444, -60.884], zoom_start=13, tiles=tile_urls[map_style] if map_style == "OpenStreetMap" else tile_urls[map_style], attr="Tiles &copy; Esri / OpenTopoMap" if map_style != "OpenStreetMap" else None)

LocateControl(
    auto_start=False,
    locateOptions={
        'setView': 'always', 
        'flyTo': True, 
        'maxZoom': 15, 
        'enableHighAccuracy': True,
        'padding': [0, 0]
    },
    keepCurrentZoomLevel=False,
    returnToPrevBounds=False,
    cacheLocation=False
).add_to(m)

Draw(draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':True}).add_to(m)

map_data = st_folium(m, use_container_width=True, height=400, key="main_map")

bbox = None
epsg_code = None
if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
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
                st.session_state.preview_image = None
                st.session_state.scene_coverage = {}  # Limpiar coverages previos
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
            # NUEVA FUNCIONALIDAD: Calcular cobertura de datos si está habilitado
            if auto_calculate_coverage and st.button("📊 Calcular % de Datos"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, scene in enumerate(all_scenes):
                    scene_id = scene.id
                    if scene_id not in st.session_state.scene_coverage:
                        try:
                            status_text.text(f"Calculando escena {idx + 1}/{len(all_scenes)}: {scene.datetime.strftime('%d/%m/%Y')}")
                            # Usar resolución baja para cálculo rápido
                            data_quick = stackstac.stack(scene, assets=[conf["assets"][0]], 
                                                        bounds_latlon=bbox, epsg=epsg_code, 
                                                        resolution=conf["res"]*4).squeeze().compute()
                            coverage = calculate_data_coverage(data_quick.values)
                            st.session_state.scene_coverage[scene_id] = coverage
                        except Exception as e:
                            st.session_state.scene_coverage[scene_id] = 0.0
                    
                    progress_bar.progress((idx + 1) / len(all_scenes))
                
                status_text.text("✅ Cálculo completado")
                st.rerun()
            
            if formato_descarga != "Video MP4":
                # Crear opciones del selectbox con información de cobertura
                scene_opts = {}
                for i, s in enumerate(all_scenes):
                    date_str = s.datetime.strftime('%d/%m/%Y')
                    cloud_str = f"Nubes: {s.properties[conf['cloud_key']]:.1f}%"
                    
                    # Agregar información de cobertura si está disponible
                    if s.id in st.session_state.scene_coverage:
                        coverage = st.session_state.scene_coverage[s.id]
                        coverage_str = f" | Datos: {coverage:.1f}%"
                        label = f"{date_str} | {cloud_str}{coverage_str}"
                    else:
                        label = f"{date_str} | {cloud_str}"
                    
                    scene_opts[label] = i
                
                idx_name = st.selectbox("Seleccionar imagen específica:", list(scene_opts.keys()))
                item = all_scenes[scene_opts[idx_name]]

                if st.session_state.current_scene_id != item.id:
                    st.session_state.preview_image = None
                    st.session_state.current_scene_id = item.id

                with st.expander("ℹ️ Información Técnica"):
                    st.write(f"**ID:** `{item.id}`")
                    st.write(f"**Plataforma:** {item.properties.get('platform', 'N/A')}")
                    st.write(f"**Nubes:** {item.properties.get(conf['cloud_key'], 0):.2f}%")
                    if item.id in st.session_state.scene_coverage:
                        st.write(f"**Cobertura de datos:** {st.session_state.scene_coverage[item.id]:.2f}%")

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    preview_btn_label = "⏳ Generando..." if st.session_state.is_generating_preview else "🖼️ Vista Previa"
                    if st.button(preview_btn_label):
                        st.session_state.is_generating_preview = True
                        st.rerun()
                    
                    if st.session_state.is_generating_preview:
                        try:
                            with st.spinner("Procesando..."):
                                data_raw = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=epsg_code, resolution=conf["res"]*2).squeeze().compute()
                                
                                # Calcular cobertura si no está calculada
                                if item.id not in st.session_state.scene_coverage:
                                    coverage = calculate_data_coverage(data_raw.sel(band=conf["assets"][0]).values)
                                    st.session_state.scene_coverage[item.id] = coverage
                                
                                img_np = np.moveaxis(data_raw.sel(band=conf["assets"][:3]).values, 0, -1)
                                img = normalize_image_robust(img_np, 2, percentil_alto, conf["scale"], conf["offset"])
                                st.session_state.preview_image = img
                        finally:
                            st.session_state.is_generating_preview = False
                            st.rerun()
                    
                    if st.session_state.preview_image is not None:
                        caption_text = f"Composición RGB: {idx_name}"
                        st.image(st.session_state.preview_image, use_container_width=True, caption=caption_text)

                with col_btn2:
                    if st.button("🚀 Descargar HD"):
                        with st.status("Procesando datos HD..."):
                            data_raw = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=epsg_code, resolution=res_final).squeeze()
                            data_final = data_raw.sel(band=conf["assets"][:3])
                            fname = f"{sat_choice.replace(' ', '_')}_{item.datetime.strftime('%Y%m%d')}_RGB"
                            if "GeoTIFF" in formato_descarga or formato_descarga == "Todos":
                                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                                    data_final.rio.to_raster(tmp.name)
                                    with open(tmp.name, 'rb') as f: st.download_button(f"📥 {fname}.tif", f.read(), f"{fname}.tif")
                            if "JPG" in formato_descarga or formato_descarga == "Todos":
                                data_np = data_final.compute().values
                                img_input = np.moveaxis(data_np, 0, -1)
                                img_8bit = normalize_image_robust(img_input, 2, percentil_alto, conf["scale"], conf["offset"])
                                buf = io.BytesIO()
                                Image.fromarray(img_8bit).save(buf, format='JPEG', quality=95)
                                st.download_button(f"📷 {fname}.jpg", buf.getvalue(), f"{fname}.jpg")

            # --- LÓGICA DE VIDEO MP4 ---
            if "Video" in formato_descarga or formato_descarga == "Todos":
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
                                status.update(label=f"Frame {processed + 1}: {date_str}...")
                                data_f = stackstac.stack(s, assets=conf["assets"], bounds_latlon=bbox, epsg=epsg_code, resolution=conf["res"]*2).squeeze().compute()
                                check_np = data_f.sel(band=conf["assets"][0]).values
                                
                                # Calcular y guardar cobertura
                                coverage = calculate_data_coverage(check_np)
                                st.session_state.scene_coverage[s.id] = coverage
                                
                                if coverage < 80: continue  # Saltar si menos del 80% tiene datos
                                
                                img_np = np.moveaxis(data_f.sel(band=conf["assets"][:3]).values, 0, -1)
                                img_8bit = normalize_image_robust(img_np, 2, percentil_alto, conf["scale"], conf["offset"])
                                pil_img = Image.fromarray(img_8bit)
                                target_w = 1000
                                h_resize = int(pil_img.height * (target_w / pil_img.width))
                                pil_img = pil_img.resize((target_w, h_resize), Image.Resampling.LANCZOS)
                                frames_list.append((s.datetime, add_text_to_image(pil_img, date_str)))
                                processed += 1
                            except: continue
                        if frames_list:
                            status.update(label="Generando video MP4...")
                            frames_list.sort(key=lambda x: x[0])
                            images_only = [np.array(f[1]) for f in frames_list]
                            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                                writer = imageio.get_writer(tmp.name, fps=video_fps, codec='libx264', quality=8)
                                for frame in images_only: writer.append_data(frame)
                                writer.close()
                                with open(tmp.name, 'rb') as f: video_bytes = f.read()
                            st.success(f"✅ Video generado: {len(images_only)} frames")
                            st.video(video_bytes, autoplay=True)
                            st.download_button("📥 Descargar Video MP4", video_bytes, "serie_temporal.mp4", mime="video/mp4")

st.markdown("---")
st.caption("Ing. Luis A. Carnaghi (lcarnaghi@gmail.com) - Creador.")
