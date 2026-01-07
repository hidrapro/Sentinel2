# 🟢 Problema: Imágenes JPG con tinte verde intenso

## 🔍 Diagnóstico del problema

### ¿Qué está pasando?

Cuando ves imágenes completamente verdes o con tintes extraños, el problema es la **normalización independiente por banda**. Aquí está el problema técnico:

### Ejemplo del error:

Imagina que tienes una imagen con estos valores reales en las 3 bandas:

```
Banda NIR (8):  valores entre 1000 - 4500
Banda SWIR (11): valores entre 800 - 3800
Banda Red (4):   valores entre 500 - 2500
```

**Normalización INCORRECTA (por banda independiente):**

```python
# Banda NIR
P2 = 1050, P98 = 4400
Stretch: (valor - 1050) / (4400 - 1050) = rango 0-1 ✓

# Banda SWIR  
P2 = 850, P98 = 3700
Stretch: (valor - 850) / (3700 - 850) = rango 0-1 ✓

# Banda Red
P2 = 550, P98 = 2400
Stretch: (valor - 550) / (2400 - 550) = rango 0-1 ✓
```

**Resultado:** Cada banda usa **diferentes valores de referencia**, causando:
- ❌ Desbalance de color
- ❌ Una banda domina sobre las otras
- ❌ Tintes verdes/azules/rojos artificiales
- ❌ Colores irreales

---

## ✅ Solución implementada

### Normalización GLOBAL (todas las bandas juntas)

En lugar de calcular percentiles por banda, se calculan sobre **todas las bandas al mismo tiempo**:

```python
# Todas las bandas juntas
valores_combinados = [valores_NIR, valores_SWIR, valores_Red]
P2_global = 900  (percentil 2 de TODOS los valores)
P98_global = 4200  (percentil 98 de TODOS los valores)

# Aplicar el MISMO stretch a todas las bandas
Banda NIR:  (valor - 900) / (4200 - 900)
Banda SWIR: (valor - 900) / (4200 - 900)  ← MISMO rango
Banda Red:  (valor - 900) / (4200 - 900)  ← MISMO rango
```

**Resultado:**
- ✅ Balance de color consistente
- ✅ No hay dominancia artificial de bandas
- ✅ Colores realistas
- ✅ Igual que el GeoTIFF

---

## 🎯 Cambios en el código

### ANTES (v4 - causaba tinte verde):

```python
def normalize_image_percentile(img_array, percentile_low=2, percentile_high=98):
    img_normalized = np.zeros_like(img_array, dtype=np.float32)
    
    for i in range(img_array.shape[2]):  # ❌ Por cada banda
        band = img_array[:, :, i]
        
        # ❌ Percentiles INDEPENDIENTES
        p_low = np.percentile(band, percentile_low)
        p_high = np.percentile(band, percentile_high)
        
        # Cada banda con su propio rango
        band_stretched = (band - p_low) / (p_high - p_low)
        img_normalized[:, :, i] = band_stretched
    
    return (img_normalized * 255).astype(np.uint8)
```

### DESPUÉS (v5 - funciona correctamente):

```python
def normalize_image_robust(img_array, percentile_low=2, percentile_high=98):
    # Máscara de valores válidos
    valid_mask = (img_array > 0) & (~np.isnan(img_array))
    
    # ✅ Percentiles GLOBALES (todas las bandas juntas)
    valid_values = img_array[valid_mask]
    p_low = np.percentile(valid_values, percentile_low)
    p_high = np.percentile(valid_values, percentile_high)
    
    # ✅ MISMO stretch para todas las bandas
    img_stretched = (img_array - p_low) / (p_high - p_low)
    img_stretched = np.clip(img_stretched, 0, 1)
    
    return (img_stretched * 255).astype(np.uint8)
```

---

## 📊 Comparación visual

### Escenario problemático:

**Imagen con mucha vegetación:**

| Método | Banda NIR | Banda SWIR | Banda Red | Resultado |
|--------|-----------|------------|-----------|-----------|
| Por banda | Stretch 0-1 | Stretch 0-1 | Stretch 0-1 | 🟢 Verde intenso |
| Global | Stretch 0-1 | Stretch 0-1 | Stretch 0-1 | ✅ Colores naturales |

La diferencia es que en el método "por banda", si la banda Red tiene valores naturalmente más bajos, se "estira" artificialmente y domina en el resultado final.

---

## 🛠️ Mejoras adicionales en v5

### 1. Modo de normalización seleccionable

En el sidebar ahora hay:

```
Método de normalización:
( ) Automático (recomendado)  ← Usa normalización global
( ) Manual                     ← Permite ajustar percentiles
```

### 2. Manejo de valores inválidos

```python
# Crear máscara para valores válidos
valid_mask = (img_array > 0) & (~np.isnan(img_array))

# Solo usar valores válidos para calcular percentiles
valid_values = img_array[valid_mask]
```

Esto elimina:
- Valores 0 (sin datos)
- NaN (errores de procesamiento)
- Valores negativos (anomalías)

### 3. Fallback seguro

Si hay muy pocos valores válidos (<100 píxeles):
```python
if np.sum(valid_mask) < 100:
    st.warning("⚠️ Pocos valores válidos. Usando normalización básica.")
    img_normalized = np.clip(img_array / 3000, 0, 1)
```

### 4. Prevención de división por cero

```python
if p_high - p_low < 1:
    p_low = np.min(valid_values)
    p_high = np.max(valid_values)
```

---

## 🎨 ¿Por qué funcionaba "al principio"?

Probablemente las primeras imágenes tenían:
- Distribución de valores más uniforme entre bandas
- Menos nubes o anomalías
- Valores más balanceados naturalmente

Las imágenes posteriores tenían:
- Más vegetación (NIR muy alto)
- Diferentes condiciones atmosféricas
- Mayor desbalance entre bandas

Con la normalización **independiente**, este desbalance se amplificaba. Con la normalización **global**, se mantiene el balance natural.

---

## 🔬 Verificación técnica

Para verificar que la v5 funciona correctamente, después de descargar un JPG:

1. Ábrelo en cualquier visor
2. Compara con el GeoTIFF en QGIS
3. Los colores deberían ser **idénticos** o muy similares

Si aún ves problemas:
1. Cambia a modo "Manual"
2. Ajusta percentiles (prueba 1-99 o 3-97)
3. Si persiste, reporta qué fecha/imagen falla

---

## 💡 Recomendaciones

### Para uso normal:
- ✅ Deja en modo "Automático"
- ✅ Percentiles default (2-98)

### Si ves saturación:
- Cambia a modo "Manual"
- Prueba percentiles 3-97 (más conservador)

### Si ves imagen muy oscura:
- Cambia a modo "Manual"  
- Prueba percentiles 1-99 (más agresivo)

### Para máxima consistencia en GIF:
- Modo "Automático" asegura que todos los frames usen el mismo método

---

## 🎯 Resultado esperado en v5

**Todas las imágenes JPG deberían:**
- ✅ Tener colores consistentes con el GeoTIFF
- ✅ No tener tintes verdes artificiales
- ✅ Verse igual que el preview
- ✅ Funcionar en la primera descarga y en todas las siguientes

---

## 🚀 Para actualizar

```bash
git add sentinel_downloader_cloud_v5_fixed.py
git commit -m "Fix: Normalización global para evitar tinte verde en JPG"
git push origin main
```

Este es un fix crítico que mejora significativamente la calidad de las imágenes JPG.
