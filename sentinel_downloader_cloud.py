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
import zipfile

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Satelites LandSat y Sentinel 2", layout="wide", page_icon="🛰️")

# --- CSS PARA MAXIMIZAR COMPACIDAD SIN PERDER VISIBILIDAD ---
st.markdown("""
    <style>
    /* Tamaño de fuente global más pequeño para que entre todo */
    html, body, [class*="st-"] {
        font-size: 0.9rem !important;
    }
    
    /* Contenedor principal: Menos padding pero con aire suficiente */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }
    
    /* Títulos: Letra más chica y márgenes controlados */
    h1 {
        font-size: 1.4rem !important;
        margin-top: -1.5rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 800 !important;
    }
    h2, h3 {
        margin-top: 0.4rem !important;
        margin-bottom: 0.2rem !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
    }
    
    /* Espaciado entre widgets: Compacto pero no asfixiante */
    [data-testid="stVerticalBlock"] {
        gap: 0.4rem !important;
    }
    
    /* Espaciado entre elementos de Streamlit */
    div.stElementContainer {
        margin-bottom: 0.2rem !important;
    }
    
    /* Compactar la barra lateral */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.4rem !important;
        padding-top: 1rem !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 0.4rem 0 !important;
    }
    
    /* Interlineado de textos */
    div[data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.2rem !important;
        line-height: 1.3 !important;
    }

    .result-text {
        display: flex;
        align-items: center;
        height: 100%;
        font-weight: bold;
        color: #2e7d32;
        font-size: 0.85rem;
    }
    
    .instruction-text {
        color: #555;
        font-style: italic;
        margin-bottom: 5px;
        display: block;
        font-size: 0.8rem;
    }
    
    .highlight-search {
        border: 2px solid #2e7d32;
        border-radius: 8px;
        padding: 1px;
        animation: pulse-green 2s infinite;
    }
    
    @keyframes pulse-green {
        0% { box-shadow: 0 0 0 0 rgba(46, 125, 50, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(46, 125, 50, 0); }
        100% { box-shadow: 0 0 0 0 rgba(46, 125, 50, 0); }
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Visualizador y descarga de recortes")

# --- INICIALIZACIÓN DE ESTADO ---
if "is_generating_preview" not in st.session_state:
    st.session_state.is_generating_preview = False
if "preview_image" not in st.session_state:
    st.session_state.preview_image = None
if "current_scene_id" not in st.session_state:
    st.session_state.current_scene_id = None
if "searching" not in st.session_state:
    st.session_state.searching = False
if "search_count" not in st.session_state:
    st.session_state.search_count = None
if "video_result" not in st.session_state:
    st.session_state.video_result = None
if "hd_file_ready" not in st.session_state:
    st.session_state.hd_file_ready = None

# --- DICCIONARIO DE CONFIGURACIÓN POR SATÉLITE ---
SAT_CONFIG = {
    "Sentinel-2": {
        "collection": "sentinel-2-l2a",
        "platform": None,
        "viz": {
            "Color Natural": ["B04", "B03", "B02"],
            "Agua-Tierra": ["B08", "B11", "B04"]
        },
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
        "viz": {
            "Color Natural": ["red", "green", "blue"],
            "Agua-Tierra": ["nir08", "swir16", "red"]
        },
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
        "viz": {
            "Color Natural": ["red", "green", "blue"],
            "Agua-Tierra": ["nir08", "swir16", "red"]
        },
        "res": 30,
        "tile_key": "landsat:wrs_path",
        "cloud_key": "eo:cloud_cover",
        "scale": 0.0000275,
        "offset": -0.2,
        "min_year": 1999,
        "max_year": datetime.now().year
    },
    "Landsat 4/5": {
        "collection": "landsat-c2-l2",
        "platform": ["landsat-4", "landsat-5"],
        "viz": {
            "Color Natural": ["red", "green", "blue"],
            "Agua-Tierra": ["nir08", "swir16", "red"]
        },
        "res": 30,
        "tile_key": "landsat:wrs_path",
        "cloud_key": "eo:cloud_cover",
        "scale": 0.0000275,
        "offset": -0.2,
        "min_year": 1982,
        "max_year": 2012
    }
}

def normalize_image_robust(img_np, sigma, perc_high, scale, offset):
    arr = (img_np * scale + offset).astype(np.float32)
    arr_valid = arr[np.isfinite(arr) & (arr > -9999)]
    if len(arr_valid) == 0: return np.zeros((*img_np.shape[:2], 3), dtype=np.uint8)
    mean_v, std_v = arr_valid.mean(), arr_valid.std()
    low_cut, high_cut = max(mean_v - sigma * std_v, 0), min(np.percentile(arr_valid, perc_high), mean_v + 3 * std_v)
    arr_clipped = np.clip((arr - low_cut) / (high_cut - low_cut + 1e-6), 0, 1)
    return (arr_clipped * 255).astype(np.uint8)

def add_text_to_image(pil_img, text):
    draw = ImageDraw.Draw(pil_img)
    font_size = int(pil_img.height * 0.04)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = right - left, bottom - top
    x, y = 15, 15
    pad = 8
    draw.rectangle([x - pad, y - pad, x + text_w + pad, y + text_h + pad], fill=(0, 0, 0, 180))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return pil_img

def check_nodata_fast(item, bbox, epsg, asset_name):
    try:
        sub = stackstac.stack(item, assets=[asset_name], bounds_latlon=bbox, epsg=epsg, resolution=200).squeeze().compute()
        arr = sub.values
        total_pix = arr.size
        valid_pix = np.sum(np.isfinite(arr) & (arr > -9000))
        return ((total_pix - valid_pix) / total_pix * 100) if total_pix > 0 else 100.0
    except:
        return 0.0

# --- BARRA LATERAL ---
with st.sidebar:
    st.markdown("### ⚙️ Configuración general")
    sat_choice = st.radio("🛰️ Misión:", list(SAT_CONFIG.keys()), index=0, horizontal=False)
    conf = SAT_CONFIG[sat_choice]
    viz_mode = st.selectbox("🎨 Visualización:", list(conf["viz"].keys()))
    selected_assets = conf["viz"][viz_mode]

    st.markdown("---")
    st.markdown("### 📅 Rango temporal")
    year_min = st.number_input("Año desde", min_value=conf["min_year"], max_value=conf["max_year"], value=conf["max_year"]-1, step=1)
    year_max = st.number_input("Año hasta", min_value=conf["min_year"], max_value=conf["max_year"], value=conf["max_year"], step=1)
    max_cloud = st.slider("☁️ Nubosidad máx (%)", 0, 100, 20)
    exclude_dates = st.text_input("Excluir fechas (DD/MM/YYYY separadas por coma):", "").split(",")
    exclude_dates = [d.strip() for d in exclude_dates if d.strip()]

    st.markdown("---")
    st.markdown("### 🖼️ Imágenes HD")
    res_mult = st.selectbox("Resolución:", [("Normal (1x)", 1), ("Alta (0.5x)", 0.5), ("Ultra (0.25x)", 0.25)], format_func=lambda x: x[0])
    res_final = int(conf["res"] * res_mult[1])
    percentil_alto = st.slider("Brillo máx (percentil)", 90, 100, 98)
    formato_descarga = st.selectbox("Formato:", ["GeoTIFF", "JPG", "KMZ", "Todos", "Video MP4"])

    if "Video" in formato_descarga or formato_descarga == "Todos":
        st.markdown("---")
        st.markdown("### 🎬 Opciones de video")
        video_fps = st.slider("FPS:", 1, 10, 3)
        video_max_images = st.slider("Máx imágenes:", 5, 50, 20)
        video_max_nodata = st.slider("Máx sin-datos (%)", 0.0, 20.0, 5.0, 0.5)

# --- MAPA INTERACTIVO ---
st.markdown("### 🗺️ Área de interés")
st.markdown('<span class="instruction-text">Dibuja un rectángulo en el mapa para buscar imágenes.</span>', unsafe_allow_html=True)

center_lat, center_lon = -34.6, -60.8
zoom_inicio = 7
m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_inicio, tiles="Esri.WorldImagery", attr="Esri")
Draw(export=False, draw_options={'polyline': False, 'polygon': False, 'circle': False, 'marker': False, 'circlemarker': False}, edit_options={'edit': False}).add_to(m)
LocateControl().add_to(m)
map_data = st_folium(m, width=None, height=400, returned_objects=["last_active_drawing"])

# --- LÓGICA DE BÚSQUEDA ---
drawing = map_data.get("last_active_drawing")
if drawing and drawing.get("geometry") and drawing["geometry"]["type"] == "Rectangle":
    coords = drawing["geometry"]["coordinates"][0]
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    bbox = (min(lons), min(lats), max(lons), max(lats))
    epsg_code = 32720 if (min(lats) + max(lats)) / 2 < 0 else 32620
    fecha_referencia = datetime(year_max, 12, 31)

    if st.session_state.searching:
        st.info("⏳ Buscando...")
    else:
        col_search, col_count = st.columns([1, 3])
        with col_search:
            search_btn_html = '<div class="highlight-search">'
            if st.button("🔍 Buscar", key="search_btn"):
                search_btn_html += '</div>'
                st.session_state.searching = True
                st.session_state.search_count = None
                st.rerun()

        if st.session_state.searching:
            try:
                catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
                query_base = {"eo:cloud_cover": {"lt": max_cloud}}
                if conf["platform"]: query_base["platform"] = {"in": conf["platform"]}
                
                search_past = catalog.search(collections=[conf["collection"]], bbox=bbox, datetime=f"{year_min}-01-01/{fecha_referencia.strftime('%Y-%m-%d')}", query=query_base)
                search_future = catalog.search(collections=[conf["collection"]], bbox=bbox, datetime=f"{fecha_referencia.strftime('%Y-%m-%d')}/{year_max}-12-31", query=query_base)
                
                all_items = list(search_past.items()) + list(search_future.items())
                if all_items:
                    with st.status("Analizando cobertura...") as status:
                        for i, item in enumerate(all_items):
                            status.update(label=f"Chequeando {i+1}/{len(all_items)}...")
                            check_asset = selected_assets[0]
                            if check_asset not in item.assets: check_asset = list(item.assets.keys())[0]
                            pct = check_nodata_fast(item, bbox, epsg_code, check_asset)
                            item.properties["custom_nodata_pct"] = pct
                    st.session_state['scenes_before'] = [i for i in all_items if i.datetime.replace(tzinfo=None) < fecha_referencia]
                    st.session_state['scenes_after'] = [i for i in all_items if i.datetime.replace(tzinfo=None) >= fecha_referencia]
                    st.session_state.search_count = len(all_items)
                else:
                    st.session_state.search_count = 0
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.search_count = 0
            finally:
                st.session_state.searching = False
                st.rerun()

        if st.session_state.search_count is not None:
            with col_count:
                if st.session_state.search_count > 0:
                    st.markdown(f'<div class="result-text">✨ {st.session_state.search_count} imágenes ({viz_mode}) encontradas.</div>', unsafe_allow_html=True)

        # --- DESPLIEGUE DE RESULTADOS ---
        if 'scenes_before' in st.session_state:
            full_pool = st.session_state['scenes_before'] + st.session_state['scenes_after']
            all_scenes = [s for s in full_pool if s.datetime.strftime('%d/%m/%Y') not in exclude_dates]
            
            # Ordenar por fecha descendente (más nueva primero)
            all_scenes.sort(key=lambda x: x.datetime, reverse=True)
            
            if all_scenes:
                if formato_descarga != "Video MP4":
                    scene_opts = {}
                    for i, s in enumerate(all_scenes):
                        pct_val = s.properties.get("custom_nodata_pct", 0.0)
                        date_str = s.datetime.strftime('%d/%m/%Y')
                        clouds = s.properties[conf['cloud_key']]
                        label = f"📅 {date_str} | ☁️ {clouds:.1f}% | ⬛ Sin Datos: {pct_val:.1f}%"
                        if pct_val > 5.0: label += " ⚠️"
                        scene_opts[label] = i
                    
                    idx_name = st.selectbox("Imagen específica:", list(scene_opts.keys()))
                    item = all_scenes[scene_opts[idx_name]]
                    
                    if st.session_state.current_scene_id != item.id:
                        st.session_state.preview_image = None
                        st.session_state.current_scene_id = item.id
                        st.session_state.hd_file_ready = None 

                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("🖼️ Vista Previa", key="prev_btn", disabled=st.session_state.is_generating_preview):
                            st.session_state.is_generating_preview = True
                            st.rerun()
                        if st.session_state.is_generating_preview:
                            try:
                                with st.spinner("Procesando..."):
                                    data_raw = stackstac.stack(item, assets=selected_assets, bounds_latlon=bbox, epsg=epsg_code, resolution=conf["res"]*2, resampling=Resampling.cubic).squeeze().compute()
                                    img_np = np.moveaxis(data_raw.sel(band=selected_assets).values, 0, -1)
                                    st.session_state.preview_image = normalize_image_robust(img_np, 2, percentil_alto, conf["scale"], conf["offset"])
                            finally:
                                st.session_state.is_generating_preview = False
                                st.rerun()
                        if st.session_state.preview_image is not None:
                            st.image(st.session_state.preview_image, use_container_width=True, caption=f"{viz_mode}")

                    with col_btn2:
                        if st.session_state.hd_file_ready is None:
                            if st.button("🚀 Generar Archivos HD", key="gen_hd_btn"):
                                with st.status("Preparando HD..."):
                                    data_raw = stackstac.stack(item, assets=selected_assets, bounds_latlon=bbox, epsg=epsg_code, resolution=res_final, resampling=Resampling.cubic).squeeze()
                                    data_final = data_raw.sel(band=selected_assets)
                                    fname = f"{sat_choice.replace(' ', '_')}_{item.datetime.strftime('%Y%m%d')}_{viz_mode.replace(' ','')}"
                                    
                                    results = {}
                                    if "GeoTIFF" in formato_descarga or formato_descarga == "Todos":
                                        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                                            data_final.rio.to_raster(tmp.name)
                                            with open(tmp.name, 'rb') as f: results['tif'] = (f.read(), f"{fname}.tif")
                                    
                                    if "JPG" in formato_descarga or formato_descarga == "Todos":
                                        img_8bit = normalize_image_robust(np.moveaxis(data_final.compute().values, 0, -1), 2, percentil_alto, conf["scale"], conf["offset"])
                                        buf = io.BytesIO()
                                        Image.fromarray(img_8bit).save(buf, format='JPEG', quality=95)
                                        results['jpg'] = (buf.getvalue(), f"{fname}.jpg")
                                    
                                    if "KMZ" in formato_descarga or formato_descarga == "Todos":
                                        data_4326 = data_final.rio.reproject("EPSG:4326", resampling=Resampling.bilinear)
                                        img_np = np.moveaxis(data_4326.compute().values, 0, -1)
                                        img_8bit_kmz = normalize_image_robust(img_np, 2, percentil_alto, conf["scale"], conf["offset"])
                                        bounds_4326 = data_4326.rio.bounds()
                                        west_kmz, south_kmz, east_kmz, north_kmz = bounds_4326
                                        buf_jpg = io.BytesIO()
                                        Image.fromarray(img_8bit_kmz).save(buf_jpg, format='JPEG', quality=95)
                                        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><GroundOverlay><n>{fname}</n><Icon><href>overlay.jpg</href></Icon><LatLonBox><north>{north_kmz}</north><south>{south_kmz}</south><east>{east_kmz}</east><west>{west_kmz}</west></LatLonBox></GroundOverlay></kml>"""
                                        kmz_buf = io.BytesIO()
                                        with zipfile.ZipFile(kmz_buf, "w") as zf:
                                            zf.writestr("doc.kml", kml_content)
                                            zf.writestr("overlay.jpg", buf_jpg.getvalue())
                                        results['kmz'] = (kmz_buf.getvalue(), f"{fname}.kmz")
                                    
                                    st.session_state.hd_file_ready = results
                                    st.rerun()
                        else:
                            st.success("✅ ¡Archivos HD listos!")
                            for key, (data, name) in st.session_state.hd_file_ready.items():
                                st.download_button(f"📥 Descargar {name}", data, name, key=f"dl_{key}", use_container_width=True)
                            if st.button("🔄 Generar otra vez", key="reset_hd"):
                                st.session_state.hd_file_ready = None
                                st.rerun()

                if "Video" in formato_descarga or formato_descarga == "Todos":
                    st.markdown("---")
                    if st.button(f"🎬 Generar Video {viz_mode}", key="gen_vid_btn"):
                        st.session_state.video_result = None
                        frames_list = []
                        pool = [s for s in sorted(all_scenes, key=lambda x: x.datetime) if s.properties.get("custom_nodata_pct", 0.0) <= video_max_nodata]
                        
                        if not pool:
                            st.error(f"Sin imágenes que cumplan (Máx: {video_max_nodata}%).")
                        else:
                            with st.status("Generando frames...") as status:
                                processed = 0
                                for s in pool:
                                    if processed >= video_max_images: break
                                    try:
                                        data_f = stackstac.stack(s, assets=selected_assets, bounds_latlon=bbox, epsg=epsg_code, resolution=conf["res"]*2, resampling=Resampling.cubic).squeeze().compute()
                                        img_np = np.moveaxis(data_f.sel(band=selected_assets).values, 0, -1)
                                        img_8bit = normalize_image_robust(img_np, 2, percentil_alto, conf["scale"], conf["offset"])
                                        pil_img = Image.fromarray(img_8bit)
                                        
                                        # MEJORA CRÍTICA PARA ANDROID:
                                        # 1. Ancho debe ser divisible por 2
                                        # 2. Alto debe ser divisible por 2
                                        target_w = 1280  # Cambio a resolución más estándar
                                        aspect_ratio = pil_img.width / pil_img.height
                                        target_h = int(target_w / aspect_ratio)
                                        
                                        # Asegurar que ambas dimensiones sean divisibles por 2
                                        target_w = (target_w // 2) * 2
                                        target_h = (target_h // 2) * 2
                                        
                                        pil_img = pil_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                                        frames_list.append((s.datetime, add_text_to_image(pil_img, s.datetime.strftime('%d/%m/%Y'))))
                                        processed += 1
                                    except: 
                                        continue
                                
                                if frames_list:
                                    status.update(label="Ensamblando...", state="running")
                                    frames_list.sort(key=lambda x: x[0])
                                    images_only = [np.array(f[1]) for f in frames_list]
                                    
                                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                                        # CONFIGURACIÓN CRÍTICA PARA COMPATIBILIDAD ANDROID:
                                        # - codec='libx264': codec H.264 (compatible con casi todos los dispositivos)
                                        # - pixelformat='yuv420p': formato de color más compatible
                                        # - macro_block_size=2: asegura dimensiones válidas
                                        # - quality=8: buena calidad (escala 1-10)
                                        # - ffmpeg_params: parámetros adicionales para mejorar compatibilidad
                                        writer = imageio.get_writer(
                                            tmp.name, 
                                            fps=video_fps, 
                                            codec='libx264',
                                            quality=8,
                                            pixelformat='yuv420p',
                                            macro_block_size=2,
                                            ffmpeg_params=[
                                                '-preset', 'medium',      # Balance velocidad/compresión
                                                '-movflags', '+faststart', # Permite reproducción antes de descarga completa
                                                '-pix_fmt', 'yuv420p'     # Redundante pero crítico para Android
                                            ]
                                        )
                                        for f in images_only: 
                                            writer.append_data(f)
                                        writer.close()
                                        
                                        with open(tmp.name, 'rb') as f: 
                                            st.session_state.video_result = f.read()
                                    
                                    status.update(label="✅ Éxito", state="complete")
                                    st.rerun()

                    if st.session_state.video_result is not None:
                        st.video(st.session_state.video_result, autoplay=True)
                        st.download_button("📥 Descargar MP4", st.session_state.video_result, "serie.mp4", key="dl_vid")

st.markdown("---")
st.caption("Ing. Luis A. Carnaghi (lcarnaghi@gmail.com) - Creador.")
