import streamlit as st
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
import xarray as xr
import math
import tempfile
import os
import requests
import tarfile
import h5py

# ==========================================
# 1. MOTOR MATEMÁTICO Y CINEMÁTICO
# ==========================================

def calcular_viento_guia(u_850, v_850, u_700, v_700, u_500, v_500):
    return (u_850 + u_700 + u_500) / 3.0, (v_850 + v_700 + v_500) / 3.0

def calcular_flujo_optico_denso(radar_tminus1, radar_t0):
    r0_8u = cv2.normalize(radar_tminus1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    r1_8u = cv2.normalize(radar_t0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(r0_8u, r1_8u, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow[..., 0], flow[..., 1]

def fusion_espaciotemporal(u_radar, v_radar, u_env, v_env, reflectividad, tau, tau_0, umbral_dbz=15):
    peso_espacial = np.clip((reflectividad - umbral_dbz) / 20.0, 0.0, 1.0)
    u_fus = (u_radar * peso_espacial) + (u_env * (1 - peso_espacial))
    v_fus = (v_radar * peso_espacial) + (v_env * (1 - peso_espacial))
    w_tau = np.exp(-tau / tau_0)
    return (w_tau * u_fus) + ((1 - w_tau) * u_env), (w_tau * v_fus) + ((1 - w_tau) * v_env)

def vector_bunkers(u_mean, v_mean, u_shr, v_shr, cizalladura_profunda, umbral_shear=20):
    if cizalladura_profunda > umbral_shear:
        D = 7.5
        modulo_shr = np.hypot(u_shr, v_shr)
        if modulo_shr > 0:
            return u_mean + (D * v_shr / modulo_shr), v_mean - (D * u_shr / modulo_shr)
    return u_mean, v_mean

def adveccion_retrograda(reflectividad, u_campo, v_campo, dt_segundos, res_metros=1000.0):
    filas, cols = reflectividad.shape
    y, x = np.mgrid[0:filas, 0:cols]
    x_origen = x - (u_campo * dt_segundos / res_metros)
    y_origen = y - (v_campo * dt_segundos / res_metros)
    coords = np.array([y_origen, x_origen])
    return map_coordinates(reflectividad, coords, order=3, mode='nearest')

def calcular_eta(lat_origen, lon_origen, lat_destino, lon_destino, u_viento, v_viento):
    R = 6371000
    phi1, phi2 = math.radians(lat_origen), math.radians(lat_destino)
    dphi, dlambda = math.radians(lat_destino - lat_origen), math.radians(lon_destino - lon_origen)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    distancia_m = R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    azimut_rad = math.atan2(y, x)
    angulo_cartesiano = (math.pi / 2.0) - azimut_rad
    vel_acercamiento = (u_viento * math.cos(angulo_cartesiano)) + (v_viento * math.sin(angulo_cartesiano))
    if vel_acercamiento <= 0:
        return f"Distancia: {distancia_m/1000:.1f} km. La tormenta no se dirige a tu ubicación."
    minutos = (distancia_m / vel_acercamiento) / 60.0
    return f"Llegará en **{minutos:.0f} minutos** (Distancia: {distancia_m/1000:.1f} km | Vel. acercamiento: {vel_acercamiento:.1f} m/s)"

# ==========================================
# 2. INTEGRACIÓN APIS
# ==========================================

def buscar_ciudad(nombre_ciudad):
    """Convierte el nombre de una ciudad en coordenadas usando Open-Meteo Geocoding."""
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": nombre_ciudad, "count": 1, "language": "es", "format": "json"}
    try:
        res = requests.get(url, params=params).json()
        if "results" in res and len(res["results"]) > 0:
            info = res["results"][0]
            region = info.get('admin1', '')
            pais = info.get('country', '')
            texto_lugar = f"{info['name']} ({region}, {pais})" if region else f"{info['name']} ({pais})"
            return info["latitude"], info["longitude"], texto_lugar
        else:
            return None, None, "Ciudad no encontrada."
    except Exception as e:
        return None, None, f"Error: {e}"

def actualizar_vientos_api(lat, lon):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=wind_speed_10m,wind_direction_10m,wind_speed_850hPa,wind_direction_850hPa,wind_speed_700hPa,wind_direction_700hPa,wind_speed_500hPa,wind_direction_500hPa&wind_speed_unit=ms"
    try:
        res = requests.get(url).json()
        act = res['current']
        def dir_vel_a_uv(vel, dir_grados):
            dir_rad = math.radians(dir_grados)
            return -vel * math.sin(dir_rad), -vel * math.cos(dir_rad)
        u_850, v_850 = dir_vel_a_uv(act['wind_speed_850hPa'], act['wind_direction_850hPa'])
        u_700, v_700 = dir_vel_a_uv(act['wind_speed_700hPa'], act['wind_direction_700hPa'])
        u_500, v_500 = dir_vel_a_uv(act['wind_speed_500hPa'], act['wind_direction_500hPa'])
        u_sfc, v_sfc = dir_vel_a_uv(act['wind_speed_10m'], act['wind_direction_10m'])
        u_shr, v_shr = u_500 - u_sfc, v_500 - v_sfc
        st.session_state.update({
            'u_850': float(u_850), 'v_850': float(v_850), 'u_700': float(u_700), 'v_700': float(v_700),
            'u_500': float(u_500), 'v_500': float(v_500), 'cizalladura': float(math.hypot(u_shr, v_shr)), 
            'u_shr': float(u_shr), 'v_shr': float(v_shr)
        })
        return True
    except Exception as e:
        st.error(f"Error con Open-Meteo: {e}")
        return False

def encontrar_radar_cercano(lat, lon):
    radares = {
        'va': (39.16, -0.25), # Valencia
        'am': (36.83, -2.67), # Almería
        'mu': (38.26, -1.18), # Murcia
        'ma': (40.17, -3.71), # Madrid
        'ba': (41.40, 1.88)   # Barcelona
    }
    distancia_minima = float('inf')
    radar_elegido = 'va'
    for codigo, coords in radares.items():
        dist = math.hypot(lat - coords[0], lon - coords[1])
        if dist < distancia_minima:
            distancia_minima = dist
            radar_elegido = codigo
    return radar_elegido

def descargar_y_procesar_aemet(api_key, lat, lon):
    codigo_radar = encontrar_radar_cercano(lat, lon)
    url_peticion = f"https://opendata.aemet.es/opendata/api/observacion/radar/local/{codigo_radar}"
    headers = {'cache-control': "no-cache"}
    
    try:
        respuesta = requests.get(url_peticion, headers=headers, params={"api_key": api_key})
        if respuesta.status_code != 200:
            return None, f"Error AEMET: {respuesta.status_code}"
            
        url_datos = respuesta.json().get("datos")
        if not url_datos:
            return None, "Radar sin datos (mantenimiento o cielo despejado)."
            
        req_file = requests.get(url_datos)
        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = os.path.join(tmpdir, "radar.tar.gz")
            with open(tar_path, 'wb') as f:
                f.write(req_file.content)
                
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(tmpdir)
                archivos = os.listdir(tmpdir)
                
            h5_file = next((f for f in archivos if f.endswith('.h5')), None)
            if not h5_file:
                return None, "El archivo AEMET descargado no contiene formato ODIM H5."
                
            with h5py.File(os.path.join(tmpdir, h5_file), 'r') as f:
                data = f['dataset1']['data1']['data'][:]
                gain = f['dataset1']['data1']['what'].attrs.get('gain', 1.0)
                offset = f['dataset1']['data1']['what'].attrs.get('offset', 0.0)
                reflectividad = (data * gain) + offset
                reflectividad = np.where(reflectividad < 0, 0, reflectividad)
                
            return reflectividad, f"Radar '{codigo_radar}' descargado con éxito."
    except Exception as e:
        return None, f"Error procesando AEMET: {str(e)}"

# ==========================================
# 3. INTERFAZ WEB (STREAMLIT)
# ==========================================

st.set_page_config(page_title="Radar Nowcasting", layout="wide")
st.title("⛈️ Predictor de Impacto de Precipitación")

# Inicialización de variables de estado (memoria de la app)
default_vars = {
    'u_850': 12.0, 'v_850': 5.0, 'u_700': 15.0, 'v_700': 8.0, 'u_500': 20.0, 'v_500': 12.0,
    'cizalladura': 25.0, 'u_shr': 18.0, 'v_shr': 10.0,
    'lat_destino': 39.47, 'lon_destino': -0.37, 'nombre_lugar': 'Valencia (Comunidad Valenciana, Spain)'
}
for k, v in default_vars.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- BARRA LATERAL ---
st.sidebar.header("1. Tu Ubicación")

# Nuevo buscador de ciudades
ciudad_input = st.sidebar.text_input("Busca tu ciudad (ej. Valencia, Pulpí):", placeholder="Escribe aquí...")
if st.sidebar.button("🔍 Buscar Ciudad", use_container_width=True):
    if ciudad_input:
        lat, lon, nombre_completo = buscar_ciudad(ciudad_input)
        if lat is not None:
            st.session_state['lat_destino'] = lat
            st.session_state['lon_destino'] = lon
            st.session_state['nombre_lugar'] = nombre_completo
            st.sidebar.success(f"Ubicación fijada: {nombre_completo}")
        else:
            st.sidebar.error("Ciudad no encontrada. Prueba con otro nombre.")

# Mostramos la ubicación seleccionada
lat_destino = st.session_state['lat_destino']
lon_destino = st.session_state['lon_destino']
st.sidebar.info(f"📍 **{st.session_state['nombre_lugar']}**\n\nLat: {lat_destino:.4f} | Lon: {lon_destino:.4f}")

st.sidebar.header("2. Clave AEMET")
api_key_aemet = st.sidebar.text_input(
    "API Key", 
    type="password",
    value="eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhem90ZWd1aW4xMUBnbWFpbC5jb20iLCJqdGkiOiIwYTRhN2Y5Ny03YTllLTQ2OTMtODhkOS0xMTkxNThmNjYyMDkiLCJleHAiOjE3OTc2Njc2MzAsImlzcyI6IkFFTUVUIiwiaWF0IjoxNzg5MDI3NjMwLCJ1c2VySWQiOiIwYTRhN2Y5Ny03YTllLTQ2OTMtODhkOS0xMTkxNThmNjYyMDkiLCJyb2xlIjoiIn0.dKqLEMMp01OKejW3viBJVzTSIXl5UGlG5FzTYUTHJV4"
)

st.sidebar.header("3. Aerograma Atmosférico")
if st.sidebar.button("🌐 Auto-completar Viento", type="primary", use_container_width=True):
    with st.spinner("Descargando ECMWF..."):
        if actualizar_vientos_api(lat_destino, lon_destino):
            st.sidebar.success("¡Datos de viento actualizados!")

col1, col2 = st.sidebar.columns(2)
u_850 = col1.number_input("U 850hPa", key='u_850', format="%.2f")
v_850 = col2.number_input("V 850hPa", key='v_850', format="%.2f")
u_700 = col1.number_input("U 700hPa", key='u_700', format="%.2f")
v_700 = col2.number_input("V 700hPa", key='v_700', format="%.2f")
u_500 = col1.number_input("U 500hPa", key='u_500', format="%.2f")
v_500 = col2.number_input("V 500hPa", key='v_500', format="%.2f")
cizalladura = st.sidebar.number_input("|V_shear| 0-6km", key='cizalladura', format="%.2f")
u_shr = col1.number_input("U Shear", key='u_shr', format="%.2f")
v_shr = col2.number_input("V Shear", key='v_shr', format="%.2f")

st.sidebar.header("4. Horizonte")
tau = st.sidebar.slider("Proyectar al futuro (minutos)", 5, 120, 30, step=5)

# --- ZONA PRINCIPAL ---
st.write("### Fuente de Datos del Radar")
metodo_datos = st.radio("Elige la fuente de datos:", [
    "Descarga Automática de AEMET",
    "Usar datos de prueba sintéticos", 
    "Subir archivo NetCDF real (.nc) manual"
])

uploaded_file = None
if metodo_datos == "Subir archivo NetCDF real (.nc) manual":
    uploaded_file = st.file_uploader("Sube el volumen", type=["nc"])
    var_name = st.text_input("Variable (ej. reflectivity)", value="reflectivity")

if st.button("Ejecutar Nowcasting", type="primary", use_container_width=True):
    with st.spinner("Procesando física de fluidos..."):
        
        # --- CARGA DE DATOS ---
        lat_tormenta, lon_tormenta = lat_destino - 0.5, lon_destino - 0.5 
        
        if metodo_datos == "Descarga Automática de AEMET":
            if not api_key_aemet:
                st.error("Falta la API Key.")
                st.stop()
                
            st.info("Conectando con AEMET y extrayendo radar más cercano...")
            lluvia_aemet, mensaje = descargar_y_procesar_aemet(api_key_aemet, lat_destino, lon_destino)
            
            if lluvia_aemet is not None:
                st.success(mensaje)
                radar_tminus1 = np.roll(lluvia_aemet, shift=-2, axis=1) 
                radar_t0 = lluvia_aemet
                lat_tormenta, lon_tormenta = lat_destino, lon_destino 
            else:
                st.error(f"Fallo en descarga: {mensaje}. Se usarán datos sintéticos.")
                metodo_datos = "Usar datos de prueba sintéticos"

        if metodo_datos == "Usar datos de prueba sintéticos" or (metodo_datos == "Subir archivo NetCDF real (.nc) manual" and uploaded_file is None):
            np.random.seed(42)
            radar_tminus1 = np.random.normal(15, 20, (150, 150))
            radar_t0 = np.roll(radar_tminus1, shift=3, axis=1) + np.random.normal(0, 2, (150, 150))
            radar_t0 = np.clip(radar_t0, 0, 55)
                
        elif metodo_datos == "Subir archivo NetCDF real (.nc) manual" and uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            try:
                ds = xr.open_dataset(tmp_path)
                lluvia = ds[var_name].values
                radar_tminus1 = lluvia[-2, :, :] if len(lluvia) > 1 else lluvia[0, :, :]
                radar_t0 = lluvia[-1, :, :]
            except Exception as e:
                st.error(f"Error procesando NetCDF: {e}")
                st.stop()
            finally:
                os.remove(tmp_path)

        # --- MOTOR FÍSICO ---
        u_guia, v_guia = calcular_viento_guia(u_850, v_850, u_700, v_700, u_500, v_500)
        u_env, v_env = vector_bunkers(u_guia, v_guia, u_shr, v_shr, cizalladura)
        
        u_radar, v_radar = calcular_flujo_optico_denso(radar_tminus1, radar_t0)
        u_eff, v_eff = fusion_espaciotemporal(
            u_radar, v_radar, 
            np.full(radar_t0.shape, u_env), np.full(radar_t0.shape, v_env), 
            radar_t0, tau, tau_0=60.0
        )
        
        radar_extrapolado = adveccion_retrograda(radar_t0, u_eff, v_eff, dt_segundos=tau*60.0)
        eta_msg = calcular_eta(lat_tormenta, lon_tormenta, lat_destino, lon_destino, u_env, v_env)

        # --- RESULTADOS ---
        st.success("Cálculos finalizados.")
        st.write("### ⏱️ Impacto Estimado (ETA)")
        st.markdown(f"> {eta_msg}")
        
        st.write(f"### Mapa de Lluvia (Proyección a +{tau} min)")
        img_radar = cv2.normalize(radar_extrapolado, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        img_color = cv2.applyColorMap(img_radar, cv2.COLORMAP_JET)
        st.image(img_color, use_container_width=True, channels="BGR")
