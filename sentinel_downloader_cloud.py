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
import geopandas as gpd
import fiona

# Habilitar soporte para KML/KMZ en fiona
fiona.drvsupport.supported_drivers['KML'] = 'rw'

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
if "epsg_code" not in st.session_state:
    st.session_state.epsg_code = None

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
    "Landsat 5": {
        "collection": "landsat-c2-l2",
        "platform": ["landsat-5"],
        "viz": {
            "Color Natural": ["red", "green", "blue"],
            "Agua-Tierra": ["nir08", "swir16", "red"]
        },
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
        "max_year": 1993
    },
    "Landsat 1-3 (MSS)": {
        "collection": "landsat-c2-l1",
        "platform": ["landsat-1", "landsat-2", "landsat-3"],
        "viz": {
            "Color Natural": ["red", "green", "blue"],
            "Agua-Tierra": ["nir08", "red", "green"]
        },
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
    lon = ((lon + 180) % 360) - 180
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
    # Aumentamos la escala de la fuente al 8% del ancho de la imagen (antes 5%)
    font_size = int(img.width * 0.08) 
    font = None
    
    # Lista ampliada y robusta para cubrir distribuciones de Linux, Windows y macOS
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc"
    ]
    
    # Intentar cargar una fuente TrueType real
    for path in font_paths:
        try: 
            font = ImageFont.truetype(path, font_size)
            break
        except: 
            continue
            
    # Red de seguridad mejorada:
    if font is None: 
        try:
            # Para versiones de Pillow >= 10.1.0 que ya permiten escalar la fuente por defecto
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            # Fallback final para versiones antiguas de Pillow (seguirá viéndose chica, debes instalar TTF en el server)
            font = ImageFont.load_default()
            
    bbox_txt = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox_txt - bbox_txt, bbox_txt - bbox_txt
    
    # Posición centrada abajo, con mayor margen
    x_pos, y_pos = (img.width - tw) // 2, img.height - th - int(font_size * 0.5)
    
    # Fondo semitransparente con padding dinámico
    padding = int(font_size * 0.2)
    draw.rectangle([(x_pos-padding, y_pos-padding), (x_pos+tw+padding, y_pos+th+padding)], fill=(0,0,0,180))
    
    # Texto
    draw.text((x_pos, y_pos), text, fill=(255, 255, 255), font=font)
    return img

def draw_gdf_on_image(img, gdf, bounds, color="#FF0000", width=3):
    """Dibuja geometrías del GeoDataFrame sobre la imagen PIL manejando coordenadas 3D"""
    if gdf is None or gdf.empty:
        return img
    
    draw = ImageDraw.Draw(img)
    xmin, ymin, xmax, ymax = bounds
    img_w, img_h = img.size
    
    dx = (xmax - xmin)
    dy = (ymax - ymin)
    if dx == 0 or dy == 0: return img
    
    scale_x = img_w / dx
    scale_y = img_h / dy
    
    for geom in gdf.geometry:
        try:
            if geom is None or geom.is_empty: continue
            
            # Unificar MultiGeometrías
            geoms = [geom] if not hasattr(geom, 'geoms') else geom.geoms
            
            for g in geoms:
                # Extraer partes lineales
                if hasattr(g, 'exterior'): # Polígono
                    parts = [g.exterior]
                elif hasattr(g, 'coords'): # LineString
                    parts = [g]
                else: continue
                
                for part in parts:
                    pixel_coords = []
                    for c in part.coords:
                        # Ignorar Z (altitud) si existe: c[0]=x, c[1]=y
                        px = (c[0] - xmin) * scale_x
                        py = (ymax - c[1]) * scale_y
                        pixel_coords.append((px, py))
                    
                    if len(pixel_coords) > 1:
                        draw.line(pixel_coords, fill=color, width=width)
        except:
            continue
    return img

def load_kmz(file):
    """Extrae el KML del KMZ y lo carga como GeoDataFrame"""
    try:
        with zipfile.ZipFile(file, 'r') as z:
            kml_filename = next(f for f in z.namelist() if f.endswith('.kml'))
            with z.open(kml_filename) as kml_file:
                gdf = gpd.read_file(kml_file, driver='KML')
                if gdf.crs is None:
                    gdf.set_crs("EPSG:4326", inplace=True)
                return gdf
    except Exception as e:
        st.sidebar.error(f"Error al cargar KMZ: {e}")
        return None

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
    
    viz_mode = st.radio("Visualización", options=list(conf["viz"].keys()), index=1, horizontal=True)
    selected_assets = conf["viz"][viz_mode]

    st.markdown("---")
    st.subheader("📍 Referencias Externas")
    uploaded_kmz = st.file_uploader("Importar KMZ (opcional)", type=["kmz"])
    kmz_gdf = None
    if uploaded_kmz:
        kmz_gdf = load_kmz(uploaded_kmz)
        if kmz_gdf is not None:
            st.success("✅ KMZ cargado")

    st.markdown("---")
    st.subheader("📅 Tiempo y Nubes")
    meses = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]
    c1, c2 = st.columns(2)
    with c1: mes_nombre = st.selectbox("Mes", meses, index=datetime.now().month - 1)
    with c2: anio = st.number_input("Año", min_value=conf["min_year"], max_value=conf["max_year"], value=conf["max_year"])
    
    mes_num = meses.index(mes_nombre) + 1
    fecha_referencia = datetime(anio, mes_num, 1)
    max_cloud = st.slider("Nubosidad máx. (%)", 0, 100, 15)
    max_search_items = st.slider("Imágenes a buscar", 2, 60, 20)
    
    st.markdown("---")
    st.subheader("⚙️ Salida")
    map_style = st.selectbox("Estilo Mapa", ["OpenStreetMap", "Satélite (Esri)", "Topográfico (OpenTopo)"])
    c3, c4 = st.columns(2)
    with c3: res_final = st.number_input("Res. (m)", value=conf["res"], min_value=10)
    with c4: percentil_alto = st.number_input("% Alto", value=98, min_value=50, max_value=100)
    formato_descarga = st.radio("Formato de descarga:", ["GeoTIFF (GIS)", "JPG (Visual)", "KMZ (Google Earth)", "Video MP4", "Todos"], horizontal=True)

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
            video_max_images = st.slider("Máx. frames", 3, 60, 15)
            video_max_nodata = st.slider("Máx. Sin Datos (%)", 0, 40, 5)
            video_overlay_kmz = st.checkbox("Superponer KMZ en video", value=True)
            video_kmz_color = st.color_picker("Color traza KMZ", "#FF0000")

# --- MAPA ---
st.subheader("1. Área de Interés (AOI)")
st.markdown('<span class="instruction-text">Click sobre la herramienta de dibujo de rectangulo AOI, icono cuadrado.</span>', unsafe_allow_html=True)

tile_urls = {"OpenStreetMap": "OpenStreetMap", "Satélite (Esri)": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Topográfico (OpenTopo)": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"}

if kmz_gdf is not None:
    center_lat = kmz_gdf.geometry.centroid.y.mean()
    center_lon = kmz_gdf.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=tile_urls[map_style] if map_style == "OpenStreetMap" else tile_urls[map_style], attr="Tiles &copy; Esri / OpenTopoMap" if map_style != "OpenStreetMap" else None)
    folium.GeoJson(kmz_gdf, name="Referencia KMZ", style_function=lambda x: {'color': 'red', 'weight': 2}).add_to(m)
else:
    m = folium.Map(location=[-35.444, -60.884], zoom_start=13, tiles=tile_urls[map_style] if map_style == "OpenStreetMap" else tile_urls[map_style], attr="Tiles &copy; Esri / OpenTopoMap" if map_style != "OpenStreetMap" else None)

LocateControl().add_to(m)
Draw(draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':True}).add_to(m)
map_data = st_folium(m, use_container_width=True, height=400, key="main_map")

bbox = None
search_allowed = True

if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons_raw = [c[0] for c in coords]
    lons = [((lon + 180) % 360) - 180 for lon in lons_raw]
    lats = [c[1] for c in coords]
    
    if max(lons) - min(lons) > 300:
        west = max([l for l in lons if l < 0])
        east = min([l for l in lons if l > 0])
        bbox = [west, min(lats), east, max(lats)]
    else:
        bbox = [min(lons), min(lats), max(lons), max(lats)]
        
    width_km = abs(bbox[2] - bbox[0]) * 111.32 * np.cos(np.radians((bbox[1]+bbox[3])/2))
    height_km = abs(bbox[3] - bbox[1]) * 110.57
    area_km2 = width_km * height_km
    
    if area_km2 > 1000:
        st.error(f"⚠️ El área seleccionada ({area_km2:.1f} km²) es demasiado grande. Máximo 1000 km².")
        search_allowed = False
    else:
        st.info(f"📏 Área seleccionada: {area_km2:.1f} km²")
        
    st.session_state.epsg_code = get_utm_epsg((min(lons)+max(lons))/2, (min(lats)+max(lats))/2)

# --- LÓGICA DE BÚSQUEDA ---
if bbox and search_allowed:
    if st.session_state.search_count is None and not st.session_state.searching:
        st.success("✅ ¡Área válida! Haz clic abajo para buscar.")
    
    col_btn, col_count = st.columns([0.2, 0.8])
    with col_btn:
        needs_highlight = st.session_state.search_count is None and not st.session_state.searching
        if needs_highlight: st.markdown('<div class="highlight-search">', unsafe_allow_html=True)
        btn_text = "Buscando..." if st.session_state.searching else "🔍 Buscar Imágenes"
        if st.button(btn_text, disabled=st.session_state.searching, key="search_btn", use_container_width=True):
            st.session_state.searching = True
            st.session_state.video_result = None
            st.session_state.hd_file_ready = None 
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
                max_items=max(1, half_items)
            )
            search_future = catalog.search(
                collections=[conf["collection"]], bbox=bbox,
                datetime=f"{fecha_referencia.isoformat()}/{f_future_end.isoformat()}",
                query=query_params, sortby=[{"field": "properties.datetime", "direction": "asc"}],
                max_items=max(1, max_search_items - half_items)
            )
            
            all_items = list(search_past.items()) + list(search_future.items())
            if all_items:
                with st.status("Analizando cobertura...") as status:
                    for i, item in enumerate(all_items):
                        status.update(label=f"Chequeando {i+1}/{len(all_items)}...")
                        check_asset = selected_assets[0]
                        if check_asset not in item.assets: check_asset = list(item.assets.keys())[0]
                        pct = check_nodata_fast(item, bbox, st.session_state.epsg_code, check_asset)
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
                                data_raw = stackstac.stack(item, assets=selected_assets, bounds_latlon=bbox, epsg=st.session_state.epsg_code, resolution=conf["res"]*2, resampling=Resampling.cubic).squeeze().compute()
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
                                data_raw = stackstac.stack(item, assets=selected_assets, bounds_latlon=bbox, epsg=st.session_state.epsg_code, resolution=res_final, resampling=Resampling.cubic).squeeze()
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
                                    kml_content = f"""<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><GroundOverlay><name>{fname}</name><Icon><href>overlay.jpg</href></Icon><LatLonBox><north>{north_kmz}</north><south>{south_kmz}</south><east>{east_kmz}</east><west>{west_kmz}</west></LatLonBox></GroundOverlay></kml>"""
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
                            video_kmz = None
                            if kmz_gdf is not None and video_overlay_kmz and st.session_state.epsg_code:
                                try:
                                    video_kmz = kmz_gdf.to_crs(epsg=st.session_state.epsg_code)
                                except:
                                    status.update(label="⚠️ Error reproyectando KMZ. Se omitirá traza.", state="running")

                            processed = 0
                            for s in pool:
                                if processed >= video_max_images: break
                                try:
                                    data_f = stackstac.stack(s, assets=selected_assets, bounds_latlon=bbox, epsg=st.session_state.epsg_code, resolution=conf["res"]*2, resampling=Resampling.cubic).squeeze().compute()
                                    img_np = np.moveaxis(data_f.sel(band=selected_assets).values, 0, -1)
                                    img_8bit = normalize_image_robust(img_np, 2, percentil_alto, conf["scale"], conf["offset"])
                                    pil_img = Image.fromarray(img_8bit)
                                    
                                    if video_kmz is not None:
                                        bounds_local = data_f.rio.bounds() # xmin, ymin, xmax, ymax
                                        pil_img = draw_gdf_on_image(pil_img, video_kmz, bounds_local, color=video_kmz_color, width=4)

                                    target_w = 720
                                    target_w = (target_w // 2) * 2
                                    h_res = (int(pil_img.height * (target_w / pil_img.width)) // 2) * 2
                                    pil_img = pil_img.resize((target_w, h_res), Image.Resampling.LANCZOS)
                                    
                                    pil_img = add_text_to_image(pil_img, s.datetime.strftime('%d/%m/%Y'))
                                    frames_list.append((s.datetime, pil_img))
                                    processed += 1
                                except Exception as e:
                                    # st.write(f"Frame ignorado por error: {e}") # Debug opcional
                                    continue
                            
                            if frames_list:
                                status.update(label="Ensamblando...", state="running")
                                frames_list.sort(key=lambda x: x[0])
                                images_only = [np.array(f[1]) for f in frames_list]
                                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                                    writer = imageio.get_writer(
                                        tmp.name, 
                                        fps=video_fps, 
                                        codec='libx264', 
                                        quality=8, 
                                        pixelformat='yuv420p',
                                        macro_block_size=2,
                                        ffmpeg_params=[
                                            '-movflags', '+faststart',
                                            '-profile:v', 'baseline',
                                            '-level', '3.0',
                                            '-maxrate', '2M',
                                            '-bufsize', '2M',
                                            '-an'
                                        ]
                                    )
                                    for f in images_only: writer.append_data(f)
                                    writer.close()
                                    with open(tmp.name, 'rb') as f: st.session_state.video_result = f.read()
                                status.update(label="✅ Éxito", state="complete")
                                st.rerun()

                if st.session_state.video_result is not None:
                    st.video(st.session_state.video_result, autoplay=True)
                    st.download_button("📥 Descargar MP4 con Traza", st.session_state.video_result, "serie_temporal_canal.mp4", key="dl_vid")

st.markdown("---")
st.caption("Ing. Luis A. Carnaghi (lcarnaghi@gmail.com) - Creador.")
