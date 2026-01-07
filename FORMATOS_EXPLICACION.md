# 📸 Formatos de descarga disponibles

## Diferencias entre GeoTIFF y JPG

### 🗺️ GeoTIFF (.tif)
**Para trabajar en software GIS (ArcGIS, QGIS, etc.)**

✅ **Ventajas:**
- Mantiene la georreferenciación (coordenadas exactas)
- Conserva valores originales de reflectancia
- Permite análisis científico (NDVI, clasificación, etc.)
- Compatible con software profesional de SIG

❌ **Limitaciones:**
- No se puede abrir en visor de fotos de Windows
- Archivos más pesados (~50-100 MB)
- Requiere software especializado

### 📷 JPG (.jpg)
**Para visualización rápida**

✅ **Ventajas:**
- Se abre con doble-click en Windows
- Archivos más livianos (~5-20 MB)
- Fácil de compartir por email/WhatsApp
- Colores optimizados para vista humana

❌ **Limitaciones:**
- Pierde la georreferenciación
- Compresión con pérdida
- No sirve para análisis científico

---

## 🎯 ¿Cuál elegir?

| Uso | Formato recomendado |
|-----|---------------------|
| Análisis en ArcGIS/QGIS | GeoTIFF |
| Ver la imagen rápidamente | JPG |
| Presentaciones PowerPoint | JPG |
| Cálculo de índices (NDVI) | GeoTIFF |
| Compartir por email | JPG |
| Archivo del proyecto | Ambos |

---

## 💡 Consejo

Si no estás seguro, descarga **Ambos**:
- El GeoTIFF para trabajar
- El JPG para compartir/presentar

El JPG se genera automáticamente optimizando el brillo y contraste para vista humana.
