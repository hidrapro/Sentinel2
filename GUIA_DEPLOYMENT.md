# Guía rápida: Subir tu proyecto a GitHub

## 1. Instalar Git (si no lo tienes)
Descarga desde: https://git-scm.com/downloads

## 2. Crear repositorio en GitHub
1. Ve a https://github.com/new
2. Nombre: `sentinel-downloader` (o el que prefieras)
3. Público o Privado (ambos funcionan con Streamlit Cloud)
4. NO agregues README, .gitignore ni licencia (ya los tienes)
5. Click "Create repository"

## 3. Subir archivos desde tu computadora

Abre la terminal/CMD en la carpeta donde tienes los archivos:

```bash
# Inicializar Git
git init

# Agregar archivos
git add sentinel_downloader_cloud.py requirements.txt README.md

# Primer commit
git commit -m "Initial commit: Sentinel-2 downloader"

# Conectar con GitHub (cambia TU_USUARIO y TU_REPO)
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git

# Subir archivos
git branch -M main
git push -u origin main
```

## 4. Desplegar en Streamlit Cloud

1. Ve a https://share.streamlit.io
2. Haz login con tu cuenta de GitHub
3. Click "New app"
4. Selecciona tu repositorio
5. Main file: `sentinel_downloader_cloud.py`
6. Click "Deploy"

## 5. ¡Listo!

Tu app estará disponible en: `https://TU_USUARIO-TU_REPO.streamlit.app`

---

## Actualizar la app después

```bash
# Hacer cambios en tu código local
# Luego subir cambios:

git add .
git commit -m "Descripción de los cambios"
git push

# Streamlit Cloud se actualizará automáticamente
```

## Troubleshooting

**Error: "remote origin already exists"**
```bash
git remote remove origin
git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
```

**Error de autenticación**
- Desde 2021, GitHub requiere Personal Access Token en lugar de contraseña
- Ve a: GitHub Settings > Developer settings > Personal access tokens
- Crea un token con permisos de "repo"
- Úsalo en lugar de tu contraseña
