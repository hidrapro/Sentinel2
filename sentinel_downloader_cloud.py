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

# --- CSS PARA MAXIMIZAR COMPACIDAD ---
st.markdown("""
    <style>
    /* Reducir márgenes del contenedor principal */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    
    /* Títulos y encabezados ultra compactos */
    h1 {
        font-size: 1.5rem !important;
        margin-top: -2.5rem !important;
        margin-bottom: 0.5rem !important;
        padding-bottom: 0 !important;
    }
    h2, h3 {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
        font-size: 1.1rem !important;
    }
    
    /* Reducir espacio entre widgets (Vertical Block Gap) */
    [data-testid="stVerticalBlock"] {
        gap: 0.2rem !important;
    }
    
    /* Reducir márgenes de cada elemento individual */
    div.stElementContainer {
        margin-bottom: 0.1rem !important;
    }
    
    /* Compactar la barra lateral */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.3rem !important;
        padding-top: 0.5rem !important;
    }
    [data-testid="stSidebar"] hr {
        margin: 0.3rem 0 !important;
    }
    
    /* Reducir interlineado en textos y párrafos */
    div[data-testid="stMarkdownContainer"] p {
        margin-bottom: 0.1rem !important;
        line-height: 1.2 !important;
    }

    .result-text {
        display: flex;
        align-items: center;
        height: 100%;
        font-weight: bold;
        color: #2e7d32;
        font-size: 0.9rem;
    }
    
    .instruction-text {
        color: #555;
        font-style: italic;
        margin-bottom: 5px;
        display: block;
        font-size: 0.85rem;
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

# --- DICCIONARIO DE CONFIGURACIÓN POR SATÉLITE ---
SAT_CONFIG = {
    "Sentinel-2": {
        "collection": "sentinel-2-l2a",
        "platform": None,
        "assets": ["B08", "B11", "B04", "B03"],
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
    utm_zone = int((lon + 180) / 6) + 1
    epsg_code = (32600 if lat >= 0 else 32700) + utm_zone
    return epsg_code

def check_nodata_fast(item, bbox, epsg, asset_name):
    """Calcula el % de pixeles negros/sin datos en el recorte exacto"""
    try:
        width_m = (bbox[2] - bbox[0]) * 111320 * np.cos(np.radians(bbox[1]))
        res_check = max(10, width_m / 50) 
        
        ds = stackstac.stack(item, assets=[asset_name], bounds_latlon=bbox, epsg=epsg, resolution=res_check).squeeze().compute()
        data = ds.values
        nodata_mask = (data <= 0) | np.isnan(data)
        percentage = (np.sum(nodata_mask) / data.size) * 100
        return float(percentage)
    except Exception:
        return 0.0

def normalize_image_robust(img_arr, p_low=2, p_high=98, scale=1.0, offset=0.0):
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
    draw = ImageDraw.Draw(img)
    font_size = int(img.width * 0.05)
    font = None
    font_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arial.ttf"]
    for path in font_paths:
        try: font = ImageFont.truetype(path, font_size); break
        except: continue
    if font is None: font = ImageFont.load_default()
    bbox_txt = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox_txt[2] - bbox_txt[0], bbox_txt[3] - bbox_txt[1]
    x_pos, y_pos = (img.width - tw) // 2, img.height - th - int(font_size * 0.3)
    draw.rectangle([(x_pos-5, y_pos-5), (x_pos+tw+5, y_pos+th+5)], fill=(0,0,0,180))
    draw.text((x_pos, y_pos), text, fill=(255, 255, 255), font=font)
    return img

# --- SIDEBAR ---
with st.sidebar:
    st.subheader("🛰️ Plataforma")
    sat_choice = st.selectbox(
        "Satélite", 
        options=list(SAT_CONFIG.keys()), 
        label_visibility="collapsed",
        format_func=lambda x: f"{x} ({SAT_CONFIG[x]['min_year']} - {SAT_CONFIG[x]['max_year']})"
    )
    conf = SAT_CONFIG[sat_choice]
    
    st.markdown("---")
    st.subheader("📅 Tiempo y Nubes")
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    c1, c2 = st.columns(2)
    with c1: mes_nombre = st.selectbox("Mes", meses, index=datetime.now().month - 1)
    with c2: anio = st.number_input("Año", min_value=conf["min_year"], max_value=conf["max_year"], value=conf["max_year"])
    
    mes_num = meses.index(mes_nombre) + 1
    fecha_referencia = datetime(anio, mes_num, 1)
    max_cloud = st.slider("Nubosidad máx. (%)", 0, 100, 15)
    max_search_items = st.slider("Imágenes a buscar", 10, 60, 20)
    
    st.markdown("---")
    st.subheader("⚙️ Salida")
    map_style = st.selectbox("Estilo Mapa", ["OpenStreetMap", "Satélite (Esri)", "Topográfico (OpenTopo)"])
    c3, c4 = st.columns(2)
    with c3: res_final = st.number_input("Res. (m)", value=conf["res"], min_value=10)
    with c4: percentil_alto = st.number_input("% Alto", value=98, min_value=50, max_value=100)
    formato_descarga = st.radio("Formato de descarga:", ["GeoTIFF (GIS)", "JPG (Visual)", "Video MP4", "Todos"], horizontal=True)

    if 'scenes_before' in st.session_state:
        with st.expander("🔍 Filtro Manual de Fechas"):
            all_candidates = st.session_state.get('scenes_before', []) + st.session_state.get('scenes_after', [])
            all_dates = sorted(list(set([s.datetime.strftime('%d/%m/%Y') for s in all_candidates])))
            exclude_dates = st.multiselect("Ignorar estas fechas:", options=all_dates)
    else:
        exclude_dates = []

    if "Video" in formato_descarga or formato_descarga == "Todos":
        with st.expander("🎬 Configuración Video"):
            video_fps = st.slider("FPS", 1, 5, 2)
            video_max_images = st.slider("Máx. frames", 3, 30, 15)
            video_max_nodata = st.slider("Máx. Sin Datos (%)", 0, 40, 5)

# --- MAPA ---
st.subheader("1. Área de Interés (AOI)")
st.markdown('<span class="instruction-text">Click sobre la herramienta de dibujo de rectangulo AOI, icono cuadrado.</span>', unsafe_allow_html=True)

tile_urls = {"OpenStreetMap": "OpenStreetMap", "Satélite (Esri)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Topográfico (OpenTopo)": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"}
m = folium.Map(location=[-35.444, -60.884], zoom_start=13, tiles=tile_urls[map_style] if map_style == "OpenStreetMap" else tile_urls[map_style], attr="Tiles &copy; Esri / OpenTopoMap" if map_style != "OpenStreetMap" else None)
LocateControl().add_to(m)
Draw(draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':True}).add_to(m)
map_data = st_folium(m, use_container_width=True, height=400, key="main_map")

bbox = None
if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    epsg_code = get_utm_epsg((min(lons)+max(lons))/2, (min(lats)+max(lats))/2)

# --- LÓGICA DE BÚSQUEDA ---
if bbox:
    if st.session_state.search_count is None and not st.session_state.searching:
        st.success("✅ ¡Área seleccionada! Haz clic abajo para buscar.")
    
    col_btn, col_count = st.columns([0.2, 0.8])
    with col_btn:
        needs_highlight = st.session_state.search_count is None and not st.session_state.searching
        if needs_highlight: st.markdown('<div class="highlight-search">', unsafe_allow_html=True)
        btn_text = "Buscando..." if st.session_state.searching else "🔍 Buscar Imágenes"
        if st.button(btn_text, disabled=st.session_state.searching, use_container_width=True):
            st.session_state.searching = True
            st.session_state.video_result = None
            st.rerun()
        if needs_highlight: st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.searching:
        try:
            catalog = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", modifier=planetary_computer.sign_inplace)
            half_items = max_search_items // 2
            query_params = {conf["cloud_key"]: {"lt": max_cloud}}
            if pool_val := conf.get("platform"): query_params["platform"] = {"in": pool_val}
            f_past_start = fecha_referencia - timedelta(days=365)
            f_future_end = fecha_referencia + timedelta(days=365)

            search_past = catalog.search(
                collections=[conf["collection"]], bbox=bbox,
                datetime=f"{f_past_start.isoformat()}/{fecha_referencia.isoformat()}",
                query=query_params, sortby=[{"field": "properties.datetime", "direction": "desc"}],
                max_items=half_items
            )
            search_future = catalog.search(
                collections=[conf["collection"]], bbox=bbox,
                datetime=f"{fecha_referencia.isoformat()}/{f_future_end.isoformat()}",
                query=query_params, sortby=[{"field": "properties.datetime", "direction": "asc"}],
                max_items=max_search_items - half_items
            )
            
            all_items = list(search_past.items()) + list(search_future.items())
            if all_items:
                with st.status("Analizando cobertura...") as status:
                    for i, item in enumerate(all_items):
                        status.update(label=f"Chequeando {i+1}/{len(all_items)}...")
                        check_asset = "B04" if "sentinel" in conf["collection"] else "red"
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
                st.markdown(f'<div class="result-text">✨ {st.session_state.search_count} imágenes equilibradas encontradas.</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-text" style="color:red">Sin resultados en el rango.</div>', unsafe_allow_html=True)

    # --- DESPLIEGUE DE RESULTADOS ---
    if 'scenes_before' in st.session_state:
        full_pool = st.session_state['scenes_before'] + st.session_state['scenes_after']
        all_scenes = [s for s in full_pool if s.datetime.strftime('%d/%m/%Y') not in exclude_dates]
        all_scenes.sort(key=lambda x: x.datetime)
        
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

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("🖼️ Vista Previa", disabled=st.session_state.is_generating_preview):
                        st.session_state.is_generating_preview = True
                        st.rerun()
                    if st.session_state.is_generating_preview:
                        try:
                            with st.spinner("Procesando..."):
                                data_raw = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=epsg_code, resolution=conf["res"]*2).squeeze().compute()
                                img_np = np.moveaxis(data_raw.sel(band=conf["assets"][:3]).values, 0, -1)
                                st.session_state.preview_image = normalize_image_robust(img_np, 2, percentil_alto, conf["scale"], conf["offset"])
                        finally:
                            st.session_state.is_generating_preview = False
                            st.rerun()
                    if st.session_state.preview_image is not None:
                        st.image(st.session_state.preview_image, use_container_width=True)

                with col_btn2:
                    if st.button("🚀 Descargar HD"):
                        with st.status("Preparando HD..."):
                            data_raw = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=epsg_code, resolution=res_final).squeeze()
                            data_final = data_raw.sel(band=conf["assets"][:3])
                            fname = f"{sat_choice.replace(' ', '_')}_{item.datetime.strftime('%Y%m%d')}"
                            if "GeoTIFF" in formato_descarga or formato_descarga == "Todos":
                                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                                    data_final.rio.to_raster(tmp.name)
                                    with open(tmp.name, 'rb') as f: st.download_button(f"📥 Descargar .tif", f.read(), f"{fname}.tif")
                            if "JPG" in formato_descarga or formato_descarga == "Todos":
                                img_8bit = normalize_image_robust(np.moveaxis(data_final.compute().values, 0, -1), 2, percentil_alto, conf["scale"], conf["offset"])
                                buf = io.BytesIO()
                                Image.fromarray(img_8bit).save(buf, format='JPEG', quality=95)
                                st.download_button(f"📷 Descargar .jpg", buf.getvalue(), f"{fname}.jpg")

            if "Video" in formato_descarga or formato_descarga == "Todos":
                st.markdown("---")
                if st.button("🎬 Generar Video MP4"):
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
                                    data_f = stackstac.stack(s, assets=conf["assets"], bounds_latlon=bbox, epsg=epsg_code, resolution=conf["res"]*2).squeeze().compute()
                                    img_np = np.moveaxis(data_f.sel(band=conf["assets"][:3]).values, 0, -1)
                                    img_8bit = normalize_image_robust(img_np, 2, percentil_alto, conf["scale"], conf["offset"])
                                    pil_img = Image.fromarray(img_8bit)
                                    target_w = 1000
                                    h_res = int(pil_img.height * (target_w / pil_img.width))
                                    pil_img = pil_img.resize((target_w, h_res), Image.Resampling.LANCZOS)
                                    frames_list.append((s.datetime, add_text_to_image(pil_img, s.datetime.strftime('%d/%m/%Y'))))
                                    processed += 1
                                except: continue
                            
                            if frames_list:
                                status.update(label="Ensamblando...", state="running")
                                frames_list.sort(key=lambda x: x[0])
                                images_only = [np.array(f[1]) for f in frames_list]
                                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                                    writer = imageio.get_writer(tmp.name, fps=video_fps, codec='libx264', quality=8)
                                    for f in images_only: writer.append_data(f)
                                    writer.close()
                                    with open(tmp.name, 'rb') as f: st.session_state.video_result = f.read()
                                status.update(label="✅ Éxito", state="complete")
                                st.rerun()

                if st.session_state.video_result is not None:
                    st.video(st.session_state.video_result, autoplay=True)
                    st.download_button("📥 Descargar MP4", st.session_state.video_result, "serie.mp4")

st.markdown("---")
st.caption("Ing. Luis A. Carnaghi (lcarnaghi@gmail.com) - Creador.")
