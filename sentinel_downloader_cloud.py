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

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Satellite HD Downloader", layout="wide", page_icon="🛰️")
st.title("🛰️ Multi-Satellite HD Downloader & Analyzer")

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

    st.header("2. Tipo de Producto")
    producto = st.selectbox(
        "Análisis",
        ["Color (Infrarrojo/Natural)", "NDVI (Vegetación)", "NDWI (Agua)"],
        help="Color: Composición de bandas (NIR-SWIR-R). NDVI: Vegetación. NDWI: Agua."
    )

    st.header("3. Filtros Temporales")
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
    st.header("4. Mapa y Salida")
    map_style = st.selectbox("Estilo del Mapa", ["OpenStreetMap", "Satélite (Esri)", "Topográfico (OpenTopo)"])
    res_final = st.number_input("Resolución descarga (m)", value=conf["res"], min_value=10)
    formato_descarga = st.radio("Formato:", ["GeoTIFF (GIS)", "JPG (Visual)", "GIF Animado", "Evolución Temporal", "Todos"])
    
    percentil_bajo, percentil_alto = 2, 98

    # --- FILTRO MANUAL DE FECHAS ---
    exclude_dates = []
    if 'scenes_before' in st.session_state and 'scenes_after' in st.session_state:
        st.markdown("---")
        st.header("5. Filtro Manual")
        all_candidates = st.session_state['scenes_before'] + st.session_state['scenes_after']
        all_dates = sorted(list(set([s.datetime.strftime('%d/%m/%Y') for s in all_candidates])))
        exclude_dates = st.multiselect("Ignorar estas fechas:", options=all_dates)

    if "GIF" in formato_descarga or "Todos" == formato_descarga:
        st.markdown("---")
        st.subheader("🎞️ Ajustes de Animación")
        gif_duration = st.slider("ms por frame", 100, 2000, 500)
        gif_max_images = st.slider("Cantidad de imágenes", 3, 30, 10)

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

def calculate_index(data, prod_type, sat_choice):
    """Calcula NDVI o NDWI basándose en el orden de assets del satélite."""
    # Mapeo según el orden definido en SAT_CONFIG
    if sat_choice == "Landsat 1-3 (MSS)":
        nir = data.sel(band="nir08")
        red = data.sel(band="red")
        green = data.sel(band="green")
    else:
        nir = data.sel(band=SAT_CONFIG[sat_choice]["assets"][0])   # nir08 / B08
        red = data.sel(band=SAT_CONFIG[sat_choice]["assets"][2])   # red / B04
        green = data.sel(band=SAT_CONFIG[sat_choice]["assets"][3]) # green / B03
    
    if prod_type == "NDVI (Vegetación)":
        return (nir - red) / (nir + red + 1e-10)
    elif prod_type == "NDWI (Agua)":
        return (green - nir) / (green + nir + 1e-10)
    return data

def add_text_to_image(img_pil, text):
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size
    fs = max(14, min(w // 20, h // 10))
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", fs)
    except: font = ImageFont.load_default()
    bb = draw.textbbox((0, 0), text, font=font)
    tw, th = bb[2]-bb[0], bb[3]-bb[1]
    m = fs // 4
    x, y = (w - tw) // 2, h - th - m * 2
    draw.rectangle([x-m, y-m, x+tw+m, y+th+m], fill=(0,0,0,180))
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
            if "Evolución Temporal" not in formato_descarga and formato_descarga != "GIF Animado":
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
                        data_raw = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=conf["res"]*2).squeeze().compute()
                        if producto == "Color (Infrarrojo/Natural)":
                            img_np = np.moveaxis(data_raw.sel(band=conf["assets"][:3]).values, 0, -1)
                        else:
                            idx_data = calculate_index(data_raw, producto, sat_choice)
                            img_np = idx_data.values
                        img = normalize_image_robust(img_np, percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                        st.image(img, use_container_width=True, caption=f"{producto}: {idx_name}")
                with col2:
                    if st.button("🚀 Descargar HD"):
                        with st.status("Procesando datos HD..."):
                            data_raw = stackstac.stack(item, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=res_final).squeeze()
                            if producto == "Color (Infrarrojo/Natural)":
                                data_final = data_raw.sel(band=conf["assets"][:3])
                            else:
                                data_final = calculate_index(data_raw, producto, sat_choice)
                            fname = f"{sat_choice.replace(' ', '_')}_{item.datetime.strftime('%Y%m%d')}_{producto[:4]}"
                            if "GeoTIFF" in formato_descarga or "Todos" == formato_descarga:
                                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                                    data_final.rio.to_raster(tmp.name)
                                    with open(tmp.name, 'rb') as f: st.download_button(f"📥 {fname}.tif", f.read(), f"{fname}.tif")
                            if "JPG" in formato_descarga or "Todos" == formato_descarga:
                                data_np = data_final.compute().values
                                img_input = np.moveaxis(data_np, 0, -1) if producto == "Color (Infrarrojo/Natural)" else data_np
                                img_8bit = normalize_image_robust(img_input, percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                                buf = io.BytesIO()
                                Image.fromarray(img_8bit).save(buf, format='JPEG', quality=95)
                                st.download_button(f"📷 {fname}.jpg", buf.getvalue(), f"{fname}.jpg")

            # --- LÓGICA DE EVOLUCIÓN TEMPORAL (GRÁFICO) ---
            if "Evolución Temporal" in formato_descarga or "Todos" == formato_descarga:
                st.markdown("---")
                st.subheader("📊 Análisis Estadístico Temporal")
                if producto == "Color (Infrarrojo/Natural)":
                    st.info("⚠️ Selecciona NDVI o NDWI en 'Tipo de Producto' para ver la evolución numérica.")
                elif st.button(f"📈 Generar Curva de {producto}"):
                    stats_data = []
                    with st.status(f"Calculando {producto} para {len(all_scenes)} fechas...") as status:
                        for s in all_scenes:
                            try:
                                date_val = s.datetime.date()
                                status.update(label=f"Procesando {date_val}...")
                                # Usamos resolución baja para velocidad de cálculo estadístico
                                data_f = stackstac.stack(s, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=conf["res"]*4).squeeze().compute()
                                idx_map = calculate_index(data_f, producto, sat_choice)
                                # Filtrar valores nulos (negros/NaNs) para no arruinar el promedio
                                valid_data = idx_map.values[idx_map.values > -1.0] # Filtro básico para índices
                                if valid_data.size > 0:
                                    mean_val = np.nanmean(valid_data)
                                    stats_data.append({"Fecha": date_val, producto: mean_val})
                            except: continue
                    
                    if stats_data:
                        df = pd.DataFrame(stats_data).set_index("Fecha")
                        st.line_chart(df)
                        st.table(df) # Mostrar tabla de valores para mayor precisión
                    else:
                        st.error("No se pudieron extraer datos estadísticos válidos.")

            # --- LÓGICA DE GIF ANIMADO ---
            if "GIF" in formato_descarga or "Todos" == formato_descarga:
                st.markdown("---")
                if st.button("🎬 Generar Animación GIF"):
                    frames_list = []
                    pool = sorted(all_scenes, key=lambda x: (abs((x.datetime.replace(tzinfo=None) - fecha_referencia).days), x.properties[conf['cloud_key']]))
                    with st.status("Generando frames...") as status:
                        processed = 0
                        for s in pool:
                            if processed >= gif_max_images: break
                            try:
                                date_str = s.datetime.strftime('%d/%m/%Y')
                                status.update(label=f"Analizando frame {processed + 1}: {date_str}...")
                                data_f = stackstac.stack(s, assets=conf["assets"], bounds_latlon=bbox, epsg=32720, resolution=conf["res"]*2).squeeze().compute()
                                check_np = data_f.sel(band=conf["assets"][0]).values
                                if np.mean(np.isnan(check_np) | (check_np <= 0)) > 0.20: continue
                                if producto == "Color (Infrarrojo/Natural)":
                                    img_np = np.moveaxis(data_f.sel(band=conf["assets"][:3]).values, 0, -1)
                                else:
                                    img_np = calculate_index(data_f, producto, sat_choice).values
                                img_8bit = normalize_image_robust(img_np, percentil_bajo, percentil_alto, conf["scale"], conf["offset"])
                                frames_list.append((s.datetime, add_text_to_image(Image.fromarray(img_8bit), date_str)))
                                processed += 1
                            except: continue
                        if frames_list:
                            frames_list.sort(key=lambda x: x[0])
                            images_only = [f[1] for f in frames_list]
                            buf = io.BytesIO()
                            images_only[0].save(buf, format='GIF', save_all=True, append_images=images_only[1:], duration=gif_duration, loop=0)
                            st.image(buf.getvalue(), caption=f"Serie {producto} - {len(images_only)} frames.")
                            st.download_button("📥 Descargar GIF", buf.getvalue(), "serie_temporal.gif")

st.markdown("---")
st.caption("Fórmulas: NDVI = (NIR-Red)/(NIR+Red) | NDWI = (Green-NIR)/(Green+NIR)")
st.caption("Ing. Luis A. Carnaghi (lcarnaghi@gmail.com) - Creador.")