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
from folium.plugins import Draw
import tempfile
from PIL import Image, ImageDraw, ImageFont
import io

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sentinel-2 HD Downloader", layout="wide", page_icon="🛰️")
st.title("🛰️ Sentinel-2 HD Downloader")

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("1. Filtros Temporales")
    
    # Selector de Mes y Año
    meses = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    col_m, col_a = st.columns(2)
    with col_m:
        mes_nombre = st.selectbox("Mes", meses, index=datetime.now().month - 1)
    with col_a:
        anio = st.number_input("Año", min_value=2015, max_value=datetime.now().year, value=datetime.now().year)
    
    mes_num = meses.index(mes_nombre) + 1
    fecha_referencia = datetime(anio, mes_num, 1)

    max_cloud = st.slider("Cobertura máxima de nubes (%)", 0, 100, 10)
    
    st.markdown("---")
    st.header("2. Configuración de Salida")
    
    st.write("📥 **Descarga directa al navegador**")
    st.write("✅ **Bandas:** 8 (NIR), 11 (SWIR), 4 (Red)")
    st.write("✅ **Resolución:** 10 metros")
    st.write("✅ **Remuestreo:** Cubic Convolution")
    
    st.markdown("---")
    st.header("3. Formato de descarga")
    formato_descarga = st.radio(
        "Selecciona formato:",
        ["GeoTIFF (para GIS)", "JPG (para visualizar)", "GIF Animado (serie temporal)", "Todos"],
        help="GeoTIFF: mantiene georreferenciación\nJPG: para visualizar\nGIF: animación de múltiples fechas"
    )
        # --- FIX CRÍTICO: Inicialización fuera del bloque IF ---
    # Esto asegura que la vista previa siempre tenga acceso a estas variables
    percentil_bajo = 2
    percentil_alto = 98
    if formato_descarga in ["JPG (para visualizar)", "GIF Animado (serie temporal)", "Todos"]:
        st.markdown("---")
        st.subheader("Ajustes de normalización")
        
        metodo_norm = st.radio(
            "Método de normalización:",
            ["Automático (recomendado)", "Manual"],
            help="Automático: usa percentiles globales\nManual: ajusta valores manualmente"
        )
        
        if metodo_norm == "Manual":
            percentil_bajo = st.slider("Corte inferior (%)", 0, 10, 2, 
                                        help="Elimina valores muy oscuros")
            percentil_alto = st.slider("Corte superior (%)", 90, 100, 98, 
                                        help="Elimina valores muy brillantes")
        else:
            # Valores fijos para modo automático
            percentil_bajo = 2
            percentil_alto = 98
            
        st.info("💡 Si ves imágenes muy verdes o saturadas, prueba el modo Manual y ajusta los percentiles")
    
    if formato_descarga in ["GIF Animado (serie temporal)", "Todos"]:
        st.markdown("---")
        st.subheader("Configuración GIF")
        gif_duration = st.slider("Duración por frame (ms)", 200, 2000, 500, 50,
                                  help="Mayor valor = animación más lenta")
        gif_max_images = st.slider("Máx. imágenes en GIF", 3, 15, 8,
                                    help="Más imágenes = archivo más grande")

# --- MAPA INTERACTIVO ---
st.subheader("1. Selecciona el área de interés (AOI)")
m = folium.Map(location=[-35.444, -60.884], zoom_start=13)
Draw(export=False, draw_options={'polyline':False, 'polygon':False, 'circle':False, 'marker':False, 'circlemarker':False, 'rectangle':True}).add_to(m)
map_data = st_folium(m, width=1200, height=450, key="main_map")

bbox = None
if map_data and map_data.get('all_drawings'):
    coords = map_data['all_drawings'][-1]['geometry']['coordinates'][0]
    lons, lats = [c[0] for c in coords], [c[1] for c in coords]
    bbox = [min(lons), min(lats), max(lons), max(lats)]
    st.success(f"Área definida: {bbox}")
else:
    st.warning("⚠️ Dibuja un rectángulo sobre el mapa.")

# --- FUNCIÓN MEJORADA PARA NORMALIZACIÓN ---
def normalize_image_robust(img_array, percentile_low=2, percentile_high=98):
    """
    Normaliza la imagen usando percentiles GLOBALES para evitar desbalance de color.
    """
    valid_mask = (img_array > 0) & (~np.isnan(img_array))
    
    if np.sum(valid_mask) < 100:
        st.warning("⚠️ Pocos valores válidos en la imagen. Usando normalización básica.")
        img_normalized = np.clip(img_array / 3000, 0, 1)
        return (img_normalized * 255).astype(np.uint8)
    
    valid_values = img_array[valid_mask]
    p_low = np.percentile(valid_values, percentile_low)
    p_high = np.percentile(valid_values, percentile_high)
    
    if p_high - p_low < 1:
        p_low = np.min(valid_values)
        p_high = np.max(valid_values)
    
    img_stretched = (img_array - p_low) / (p_high - p_low)
    img_stretched = np.clip(img_stretched, 0, 1)
    
    return (img_stretched * 255).astype(np.uint8)

def normalize_image_per_band(img_array, percentile_low=2, percentile_high=98):
    """
    Normalización independiente por banda.
    """
    img_normalized = np.zeros_like(img_array, dtype=np.float32)
    for i in range(img_array.shape[2]):
        band = img_array[:, :, i]
        valid_mask = (band > 0) & (~np.isnan(band))
        if np.sum(valid_mask) < 50:
            img_normalized[:, :, i] = band / 3000
            continue
        valid_values = band[valid_mask]
        p_low = np.percentile(valid_values, percentile_low)
        p_high = np.percentile(valid_values, percentile_high)
        band_stretched = (band - p_low) / (p_high - p_low)
        band_stretched = np.clip(band_stretched, 0, 1)
        img_normalized[:, :, i] = band_stretched
    return (img_normalized * 255).astype(np.uint8)

def add_text_to_image(img_pil, text, position='bottom'):
    """
    Agrega texto (fecha) a una imagen PIL con escalado mejorado para imágenes pequeñas.
    """
    draw = ImageDraw.Draw(img_pil)
    
    # Calcular tamaño de fuente adaptativo basado en el ancho y alto
    img_width, img_height = img_pil.size
    # Reducimos el mínimo de 20 a 12 para que no sea gigante en previews pequeñas
    font_size = max(12, min(img_width // 25, img_height // 12))
    
    try:
        # Intentar usar fuente del sistema
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        # Fallback a fuente default
        font = ImageFont.load_default()
    
    # Obtener dimensiones del texto
    bbox_text = draw.textbbox((0, 0), text, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]
    
    # Ajustar paddings proporcionales
    padding_edge = max(4, img_height // 40)
    bg_margin = max(2, font_size // 6)
    
    # Posición del texto (centrado horizontalmente)
    x = (img_width - text_width) // 2
    
    if position == 'bottom':
        y = img_height - text_height - padding_edge - bg_margin
    else:  # top
        y = padding_edge + bg_margin
    
    # Dibujar fondo semi-transparente proporcional al texto
    draw.rectangle(
        [x - bg_margin, y - bg_margin, 
         x + text_width + bg_margin, y + text_height + bg_margin],
        fill=(0, 0, 0, 160)
    )
    
    # Dibujar texto
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return img_pil

# --- LÓGICA DE BÚSQUEDA ---
if bbox:
    st.subheader(f"2. Imágenes cerca de {mes_nombre} {anio}")
    
    if st.button("🔍 Buscar (10 antes y 10 después)"):
        with st.spinner("Consultando catálogo..."):
            try:
                catalog = pystac_client.Client.open(
                    "https://planetarycomputer.microsoft.com/api/stac/v1",
                    modifier=planetary_computer.sign_inplace,
                )
                
                search_before = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=f"2015-06-01/{fecha_referencia.isoformat()}",
                    max_items=10,
                    query={"eo:cloud_cover": {"lt": max_cloud}},
                    sortby=[{"field": "properties.datetime", "direction": "desc"}]
                )
                
                search_after = catalog.search(
                    collections=["sentinel-2-l2a"],
                    bbox=bbox,
                    datetime=f"{fecha_referencia.isoformat()}/{datetime.now().isoformat()}",
                    max_items=10,
                    query={"eo:cloud_cover": {"lt": max_cloud}},
                    sortby=[{"field": "properties.datetime", "direction": "asc"}]
                )
                
                scenes = list(search_before.items()) + list(search_after.items())
                scenes.sort(key=lambda x: x.datetime)
                
                if scenes:
                    st.session_state['scenes_list'] = scenes
                    st.success(f"✅ Encontradas {len(scenes)} imágenes")
                else:
                    st.error("No se encontraron imágenes con esos filtros.")
            except Exception as e:
                st.error(f"Error: {e}")

    if 'scenes_list' in st.session_state:
        scenes = st.session_state['scenes_list']
        
        if formato_descarga != "GIF Animado (serie temporal)":
            st.subheader("3. Descarga de imagen individual")
            scene_options = {
                f"{s.datetime.strftime('%d/%m/%Y')} | Nubes: {s.properties['eo:cloud_cover']:.1f}% | Tile: {s.properties.get('s2:mgrs_tile')}": i 
                for i, s in enumerate(scenes)
            }
            selected_label = st.selectbox("Elegir imagen:", list(scene_options.keys()))
            item = scenes[scene_options[selected_label]]

            with st.expander("🖼️ Vista Previa Rápida", expanded=True):
                if st.button("Generar Preview"):
                    try:
                        prev = stackstac.stack(item, assets=["B08", "B11", "B04"], bounds_latlon=bbox, epsg=32720, resolution=60).squeeze().compute()
                        img = np.moveaxis(prev.values, 0, -1)
                        img_preview = normalize_image_robust(img, percentil_bajo, percentil_alto)
                        st.image(img_preview, use_container_width=True, caption="Preview con normalización robusta")
                    except Exception as e:
                        st.error(f"Error preview: {e}")

            if st.button("🚀 Descargar Imagen HD (10m)"):
                try:
                    with st.status("Procesando descarga HD...", expanded=True) as status:
                        data = stackstac.stack(
                            item, assets=["B08", "B11", "B04"], bounds_latlon=bbox, 
                            epsg=32720, resolution=10, resampling=Resampling.cubic
                        ).squeeze()
                        
                        data = data.assign_coords(band=["8", "11", "4"])
                        fname_base = f"T{item.properties.get('s2:mgrs_tile')}_{item.datetime.strftime('%Y%m%d')}"
                        img_array = np.moveaxis(data.values, 0, -1)
                        
                        status.update(label="Generando archivos...", state="running")
                        
                        if formato_descarga in ["GeoTIFF (para GIS)", "Todos"]:
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_tif:
                                data.rio.to_raster(tmp_tif.name)
                                with open(tmp_tif.name, 'rb') as f:
                                    tif_bytes = f.read()
                                os.unlink(tmp_tif.name)
                            
                            st.download_button(
                                label=f"📥 Descargar {fname_base}.tif",
                                data=tif_bytes, file_name=f"{fname_base}.tif", mime="image/tiff"
                            )
                        
                        if formato_descarga in ["JPG (para visualizar)", "Todos"]:
                            img_8bit = normalize_image_robust(img_array, percentil_bajo, percentil_alto)
                            img_pil = Image.fromarray(img_8bit)
                            jpg_buffer = io.BytesIO()
                            img_pil.save(jpg_buffer, format='JPEG', quality=95)
                            st.download_button(
                                label=f"📷 Descargar {fname_base}.jpg",
                                data=jpg_buffer.getvalue(), file_name=f"{fname_base}.jpg", mime="image/jpeg"
                            )
                        
                        status.update(label=f"¡Listo para descargar!", state="complete")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        else:
            st.subheader("3. Generar GIF Animado (Serie Temporal)")
            max_imgs = gif_max_images if 'gif_max_images' in locals() else 15
            duration = gif_duration if 'gif_duration' in locals() else 500
            available_scenes = scenes[:max_imgs]
            
            st.info(f"📊 Se usarán las primeras **{len(available_scenes)}** imágenes encontradas")
            
            with st.expander("📅 Fechas incluidas en el GIF", expanded=True):
                cols = st.columns(4)
                for idx, scene in enumerate(available_scenes):
                    with cols[idx % 4]:
                        st.write(f"{idx+1}. {scene.datetime.strftime('%d/%m/%Y')}")
            
            if st.button("🎬 Generar GIF Animado"):
                try:
                    with st.status(f"Procesando {len(available_scenes)} imágenes...", expanded=True) as status:
                        frames = []
                        for idx, scene in enumerate(available_scenes):
                            status.update(label=f"Procesando imagen {idx+1}/{len(available_scenes)}...", state="running")
                            data = stackstac.stack(
                                scene, assets=["B08", "B11", "B04"], bounds_latlon=bbox, 
                                epsg=32720, resolution=30, resampling=Resampling.cubic
                            ).squeeze().compute()
                            img_array = np.moveaxis(data.values, 0, -1)
                            img_8bit = normalize_image_robust(img_array, percentil_bajo, percentil_alto)
                            img_pil = Image.fromarray(img_8bit)
                            
                            fecha_texto = scene.datetime.strftime('%d/%m/%Y')
                            img_pil = add_text_to_image(img_pil, fecha_texto, position='bottom')
                            frames.append(img_pil)
                        
                        gif_buffer = io.BytesIO()
                        frames[0].save(
                            gif_buffer, format='GIF', save_all=True, append_images=frames[1:],
                            duration=duration, loop=0, optimize=True
                        )
                        gif_bytes = gif_buffer.getvalue()
                        status.update(label="¡GIF generado exitosamente!", state="complete")
                        st.image(gif_bytes, caption="Preview del GIF")
                        
                        fname_gif = f"Serie_Temporal_{fecha_referencia.strftime('%Y%m')}.gif"
                        st.download_button(
                            label=f"🎬 Descargar {fname_gif}",
                            data=gif_bytes, file_name=fname_gif, mime="image/gif"
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

st.markdown("---")
st.caption(f"9 de Julio, Argentina | Escala de texto corregida | GIF animado de series temporales")