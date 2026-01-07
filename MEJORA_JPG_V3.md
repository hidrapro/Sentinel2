# 🎨 Mejora de calidad JPG - Versión 3

## ❌ Problema en versión anterior

La imagen JPG se veía **muy lavada y amarillenta** porque:

1. **Normalización fija**: Dividía todos los valores por 3500
2. **No adaptativa**: No consideraba la distribución real de valores en cada imagen
3. **Sin stretch de contraste**: Los valores se comprimían en un rango pequeño

### Ejemplo del problema:
- Si los valores reales van de 500 a 2000
- Al dividir por 3500: rango resultante 0.14 - 0.57
- Se pierde todo el contraste ❌

---

## ✅ Solución implementada: Normalización por percentiles

### Cómo funciona ahora:

1. **Análisis por banda**: Cada banda (NIR, SWIR, Red) se procesa independientemente

2. **Corte de extremos**: 
   - Percentil 2% (default): elimina valores muy oscuros (sombras, nubes)
   - Percentil 98% (default): elimina valores muy brillantes (saturación, reflejos)

3. **Stretch lineal adaptativo**:
   ```python
   valor_normalizado = (valor - percentil_2) / (percentil_98 - percentil_2)
   ```

4. **Conversión a 8 bits**: Resultado final 0-255 con máximo contraste

### Resultado:
- ✅ Colores vibrantes (como en el GeoTIFF)
- ✅ Buen contraste
- ✅ Sin valores extremos que distorsionen la imagen
- ✅ Adaptativo a cada escena

---

## 🎛️ Controles ajustables

En el **sidebar** puedes modificar:

### Percentil Inferior (default: 2%)
- **Más bajo (0-1%)**: Imagen más oscura, más contraste
- **Más alto (3-5%)**: Elimina más sombras, imagen más clara

### Percentil Superior (default: 98%)
- **Más bajo (95-97%)**: Elimina más saturación, colores más suaves
- **Más alto (99-100%)**: Mantiene más brillo, puede verse saturado

---

## 📊 Comparación técnica

| Aspecto | V2 (mala) | V3 (mejorada) |
|---------|-----------|---------------|
| Método | División fija /3500 | Percentiles adaptativos |
| Contraste | Bajo, lavado | Alto, natural |
| Adaptabilidad | Ninguna | Por escena |
| Colores | Amarillentos | Fieles al GeoTIFF |
| Ajustable | No | Sí (sliders) |

---

## 💡 Recomendaciones de uso

### Para imágenes normales:
- Percentil bajo: **2%**
- Percentil alto: **98%**

### Para imágenes con muchas nubes:
- Percentil bajo: **5%** (elimina más sombras de nubes)
- Percentil alto: **95%** (elimina brillos de nubes)

### Para máximo contraste:
- Percentil bajo: **1%**
- Percentil alto: **99%**

### Para imagen más suave:
- Percentil bajo: **3%**
- Percentil alto: **97%**

---

## 🔬 Ejemplo técnico

**Imagen con valores reales:**
```
Banda NIR: min=800, max=4200
Banda SWIR: min=600, max=3800
Banda Red: min=400, max=2500
```

**V2 (malo):**
```
NIR:  800/3500 = 0.23 → 4200/3500 = 1.20 (saturado) ❌
SWIR: 600/3500 = 0.17 → 3800/3500 = 1.09 (saturado) ❌
Red:  400/3500 = 0.11 → 2500/3500 = 0.71 ❌
```

**V3 (bueno):**
```
NIR:  P2=850, P98=4000 → stretch [0, 1] → [0, 255] ✅
SWIR: P2=650, P98=3500 → stretch [0, 1] → [0, 255] ✅
Red:  P2=450, P98=2300 → stretch [0, 1] → [0, 255] ✅
```

Resultado: **Máximo uso del rango 0-255 = mejor contraste**

---

## 🎯 Ahora la imagen JPG se verá:

- **Como el GeoTIFF** en términos de color y contraste
- **Compatible con Windows** (doble-click para abrir)
- **Optimizada para vista humana** (sin ruido, sin saturación)
- **Ajustable** según tus preferencias

¡Ya no más imágenes amarillentas! 🎉
