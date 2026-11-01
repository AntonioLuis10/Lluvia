"""
Radar Nowcasting (Streamlit)

Flujo:
  1. Frames de radar reales (RainViewer), sintéticos o NetCDF propio.
  2. Flujo óptico (Farneback) entre frames -> velocidad en m/s.
  3. Fusión con el viento ambiente (Open-Meteo, niveles de presión).
  4. Advección semi-lagrangiana hacia atrás, paso a paso.
  5. ETA calculado sobre el propio campo de reflectividad proyectado.

Convenciones:
  - u: positivo hacia el ESTE (m/s). v_norte: positivo hacia el NORTE (m/s).
  - En imágenes las filas crecen hacia el SUR, así que internamente se usa
    v_sur = -v_norte para advectar.
"""
import io
import math
import os
import tempfile
from datetime import datetime, timezone

import cv2
import numpy as np
import requests
import streamlit as st
import xarray as xr
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo("Europe/Madrid")
except Exception:  # sin base de datos de zonas horarias
    TZ = timezone.utc

TIMEOUT = 20          # s, para todas las peticiones HTTP
PASO_MIN = 5          # paso temporal de la proyección
TAU_0_MIN = 60.0      # constante de decaimiento del peso del flujo óptico
V_MAX = 40.0          # m/s, límite de cordura para el flujo óptico
UMBRAL_FUSION_DBZ = 15.0
RV_API = "https://api.rainviewer.com/public/weather-maps.json"

# ==========================================
# 1. DECODIFICACIÓN DE RAINVIEWER (paleta "Universal Blue")
# ==========================================
# Desde 2026 la API gratuita sólo sirve la paleta Universal Blue (color=2).
# Cada color corresponde a una fila de la tabla oficial: dBZ = fila - 32.
# Aquí van las filas 22..96 (dBZ -10..64) y el blanco (>= 65 dBZ).
_PALETA_RGB = [
    (99, 97, 89), (102, 99, 90), (105, 102, 92), (108, 104, 93), (111, 107, 95),
    (114, 110, 97), (117, 112, 98), (120, 115, 100), (124, 117, 101), (127, 120, 103),
    (130, 123, 105), (133, 125, 106), (136, 128, 108), (139, 130, 109), (142, 133, 111),
    (146, 136, 113), (158, 147, 117), (170, 158, 121), (182, 169, 126), (194, 180, 130),
    (206, 192, 135), (210, 196, 139), (214, 200, 143), (218, 204, 147), (222, 208, 151),
    (136, 221, 238), (108, 209, 235), (81, 197, 232), (54, 186, 229), (27, 174, 226),
    (0, 163, 224), (0, 154, 213), (0, 145, 202), (0, 136, 191), (0, 127, 180),
    (0, 119, 170), (0, 112, 163), (0, 105, 156), (0, 98, 149), (0, 91, 142),
    (0, 85, 136), (0, 81, 128), (0, 78, 120), (0, 74, 112), (0, 71, 104),
    (255, 238, 0), (255, 224, 0), (255, 210, 0), (255, 197, 0), (255, 183, 0),
    (255, 170, 0), (255, 159, 0), (255, 149, 0), (255, 139, 0), (255, 129, 0),
    (255, 68, 0), (242, 54, 0), (230, 40, 0), (217, 27, 0), (205, 13, 0),
    (193, 0, 0), (168, 0, 0), (143, 0, 0), (118, 0, 0), (93, 0, 0),
    (255, 170, 255), (255, 159, 255), (255, 149, 255), (255, 139, 255), (255, 129, 255),
    (255, 119, 255), (255, 108, 255), (255, 98, 255), (255, 88, 255), (255, 78, 255),
    (255, 255, 255),
]
_DBZ_PALETA = np.arange(len(_PALETA_RGB), dtype=np.float32) - 10.0
_ARBOL_PALETA = cKDTree(np.array(_PALETA_RGB, dtype=np.float64))


def tile_a_dbz(rgba):
    """RGBA (H,W,4) uint8 -> reflectividad en dBZ (0 = sin eco)."""
    dbz = np.zeros(rgba.shape[:2], dtype=np.float32)
    con_eco = rgba[..., 3] > 0
    if con_eco.any():
        _, idx = _ARBOL_PALETA.query(rgba[..., :3][con_eco].astype(np.float64))
        dbz[con_eco] = _DBZ_PALETA[idx]
    dbz[dbz < 5.0] = 0.0  # ruido / eco débil
    return dbz


def km_por_pixel(lat, z):
    return 40075.017 * math.cos(math.radians(lat)) / (2 ** z) / 256.0


@st.cache_data(ttl=120, show_spinner=False)
def rv_lista_frames():
    r = requests.get(RV_API, timeout=TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return j["host"], j["radar"]["past"]


@st.cache_data(ttl=900, show_spinner=False)
def rv_tile_dbz(host, path, lat, lon, z):
    # Este endpoint devuelve un tile de 256 px CENTRADO en (lat, lon).
    url = f"{host}{path}/256/{z}/{lat:.4f}/{lon:.4f}/2/0_0.png"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    rgba = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
    return tile_a_dbz(rgba)


def cargar_rainviewer(lat, lon, z, n_frames=3):
    host, pasados = rv_lista_frames()
    if len(pasados) < 2:
        raise RuntimeError("RainViewer no devolvió suficientes frames.")
    sel = pasados[-n_frames:]
    campos = [rv_tile_dbz(host, f["path"], round(lat, 4), round(lon, 4), z) for f in sel]
    tiempos = [int(f["time"]) for f in sel]
    return {
        "campos": campos,
        "dt_s": [float(b - a) for a, b in zip(tiempos[:-1], tiempos[1:])],
        "kmpp": km_por_pixel(lat, z),
        "px": campos[0].shape[1] // 2,
        "py": campos[0].shape[0] // 2,
        "t_ref": tiempos[-1],
        "origen": f"RainViewer ({len(campos)} frames, zoom {z})",
    }


# ==========================================
# 2. OTROS ORÍGENES DE DATOS
# ==========================================
def datos_sinteticos():
    """Tres frames con una célula que se mueve hacia el este a ~48 km/h."""
    n = 256
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)

    def campo(cx, cy):
        c = 50 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 12 ** 2))
        c += 38 * np.exp(-((xx - (cx - 35)) ** 2 + (yy - (cy + 22)) ** 2) / (2 * 9 ** 2))
        c += 30 * np.exp(-((xx - (cx + 25)) ** 2 + (yy - (cy - 20)) ** 2) / (2 * 7 ** 2))
        c[c < 5] = 0
        return c.astype(np.float32)

    dx, dy = 8.0, 1.0  # px por frame (10 min)
    cx0, cy0 = 50.0, 118.0
    campos = [campo(cx0 + k * dx, cy0 + k * dy) for k in range(3)]
    return {
        "campos": campos, "dt_s": [600.0, 600.0], "kmpp": 1.0,
        "px": n // 2, "py": n // 2, "t_ref": None,
        "origen": "Datos sintéticos (célula hacia el este, 1 km/px)",
    }


def cargar_netcdf(archivo, var_name, res_km, dt_min, px, py):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
        tmp.write(archivo.read())
        ruta = tmp.name
    try:
        with xr.open_dataset(ruta) as ds:
            arr = np.squeeze(np.asarray(ds[var_name].values, dtype=np.float32))
    finally:
        os.remove(ruta)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError(f"Se esperaba (tiempo, y, x) con >=2 frames; llegó {arr.shape}.")
    arr = np.clip(np.nan_to_num(arr, nan=0.0), 0.0, None)
    campos = list(arr[-3:])
    return {
        "campos": campos, "dt_s": [dt_min * 60.0] * (len(campos) - 1),
        "kmpp": float(res_km), "px": int(px), "py": int(py), "t_ref": None,
        "origen": f"NetCDF ({var_name}, {len(campos)} frames)",
    }


def obtener_vientos_open_meteo(lat, lon):
    """Viento actual en 850/700/500 hPa y superficie (hourly, UTC)."""
    niveles = ["850hPa", "700hPa", "500hPa"]
    hourly = ["wind_speed_10m", "wind_direction_10m"]
    for n in niveles:
        hourly += [f"wind_speed_{n}", f"wind_direction_{n}"]
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat, "longitude": lon, "hourly": ",".join(hourly),
            "wind_speed_unit": "ms", "timezone": "UTC", "forecast_days": 2,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    h = r.json()["hourly"]
    ahora = datetime.now(timezone.utc).replace(tzinfo=None)
    tiempos = [datetime.strptime(t, "%Y-%m-%dT%H:%M") for t in h["time"]]
    pasadas = [i for i, t in enumerate(tiempos) if t <= ahora]
    idx = pasadas[-1] if pasadas else 0

    def uv(sufijo):
        vel, dire = h[f"wind_speed_{sufijo}"][idx], h[f"wind_direction_{sufijo}"][idx]
        if vel is None or dire is None:
            raise ValueError(f"Open-Meteo no devolvió datos de {sufijo}.")
        a = math.radians(float(dire))  # dirección meteorológica: DE DONDE sopla
        return -float(vel) * math.sin(a), -float(vel) * math.cos(a)

    u850, v850 = uv("850hPa")
    u700, v700 = uv("700hPa")
    u500, v500 = uv("500hPa")
    usfc, vsfc = uv("10m")
    return {
        "u_850": u850, "v_850": v850, "u_700": u700, "v_700": v700,
        "u_500": u500, "v_500": v500,
        "u_shr": u500 - usfc, "v_shr": v500 - vsfc,
    }


# ==========================================
# 3. MOTOR FÍSICO
# ==========================================
def viento_guia(u850, v850, u700, v700, u500, v500):
    return (u850 + u700 + u500) / 3.0, (v850 + v700 + v500) / 3.0


def vector_bunkers(u_mean, v_mean, u_shr, v_shr, umbral_shear=20.0, D=7.5):
    """Movimiento de la supercélula (derecha) si la cizalladura es fuerte."""
    modulo = math.hypot(u_shr, v_shr)
    if modulo > umbral_shear:
        return u_mean + D * v_shr / modulo, v_mean - D * u_shr / modulo
    return u_mean, v_mean


def _a_uint8(dbz):
    # Escala FIJA (no min-max por frame): mantiene la constancia de brillo.
    return (np.clip(dbz, 0, 60) / 60.0 * 255.0).astype(np.uint8)


def flujo_optico(dbz_a, dbz_b, kmpp, dt_s):
    """Farneback entre dos frames -> (u_este, v_sur) en m/s."""
    a = cv2.GaussianBlur(_a_uint8(dbz_a), (5, 5), 0)
    b = cv2.GaussianBlur(_a_uint8(dbz_b), (5, 5), 0)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.1, 0)
    factor = kmpp * 1000.0 / dt_s  # px/frame -> m/s
    return flow[..., 0] * factor, flow[..., 1] * factor


def flujo_medio(campos, dt_s, kmpp):
    us, vs = [], []
    for i in range(len(campos) - 1):
        u, v = flujo_optico(campos[i], campos[i + 1], kmpp, dt_s[i])
        us.append(u)
        vs.append(v)
    u, v = np.mean(us, axis=0), np.mean(vs, axis=0)
    u, v = gaussian_filter(u, 2.0), gaussian_filter(v, 2.0)
    mag = np.hypot(u, v)
    f = np.minimum(1.0, V_MAX / np.maximum(mag, 1e-6))
    return (u * f).astype(np.float32), (v * f).astype(np.float32)


def fusion_espaciotemporal(u_r, v_r, u_e, v_e, refl, lead_min):
    """Mezcla radar/ambiente: más radar donde hay eco y a corto plazo."""
    w_esp = np.clip((refl - UMBRAL_FUSION_DBZ) / 20.0, 0.0, 1.0)
    w = w_esp * math.exp(-lead_min / TAU_0_MIN)
    return w * u_r + (1 - w) * u_e, w * v_r + (1 - w) * v_e


def avanzar(campo, u, v_sur, dt_s, res_m):
    """Advección semi-lagrangiana hacia atrás (u hacia +x, v_sur hacia +fila)."""
    filas, cols = campo.shape
    yy, xx = np.mgrid[0:filas, 0:cols].astype(np.float32)
    coords = np.array([yy - v_sur * dt_s / res_m, xx - u * dt_s / res_m])
    return map_coordinates(campo, coords, order=1, mode="constant", cval=0.0)


def pronosticar(dbz0, u_r, v_r, u_env, v_env_sur, kmpp, horizonte_min):
    """Devuelve {minutos: campo} y el campo de velocidades inicial fusionado."""
    res_m = kmpp * 1000.0
    campo, ur, vr = dbz0.copy(), u_r.copy(), v_r.copy()
    frames = {0: dbz0.copy()}
    u_ini = v_ini = None
    for t in range(0, horizonte_min, PASO_MIN):
        u, v = fusion_espaciotemporal(ur, vr, u_env, v_env_sur, campo, t + PASO_MIN / 2.0)
        if u_ini is None:
            u_ini, v_ini = u, v
        campo = np.clip(avanzar(campo, u, v, PASO_MIN * 60.0, res_m), 0.0, None)
        ur = avanzar(ur, u, v, PASO_MIN * 60.0, res_m)  # el flujo viaja con la célula
        vr = avanzar(vr, u, v, PASO_MIN * 60.0, res_m)
        frames[t + PASO_MIN] = campo
    return frames, u_ini, v_ini


def calcular_eta(frames, kmpp, px, py, radio_km, umbral_dbz):
    """Primer instante en que el máximo en un disco alrededor del punto supera el umbral."""
    r = max(1, int(round(radio_km / kmpp)))
    h, w = frames[0].shape
    y0, y1, x0, x1 = max(0, py - r), min(h, py + r + 1), max(0, px - r), min(w, px + r + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disco = (yy - py) ** 2 + (xx - px) ** 2 <= r * r
    for t in sorted(frames):
        m = float(frames[t][y0:y1, x0:x1][disco].max())
        if m >= umbral_dbz:
            return t, m
    return None, 0.0


# ==========================================
# 4. VISUALIZACIÓN
# ==========================================
def render(dbz, kmpp, px, py, escala=3):
    h, w = dbz.shape
    fondo = np.full((h, w, 3), (40, 34, 30), np.uint8)  # BGR
    color = cv2.applyColorMap(_a_uint8(dbz), cv2.COLORMAP_JET)
    fondo[dbz >= 10] = color[dbz >= 10]
    img = cv2.resize(fondo, None, fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR)
    cx, cy = int(round((px + 0.5) * escala)), int(round((py + 0.5) * escala))
    for km in (25, 50, 100):
        rad = int(round(km / kmpp * escala))
        if rad < max(h, w) * escala:
            cv2.circle(img, (cx, cy), rad, (140, 140, 140), 1, cv2.LINE_AA)
            cv2.putText(img, f"{km} km", (cx + 4, cy - rad + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def hora_local(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(TZ).strftime("%H:%M")


def rumbo_hacia(u, v_norte):
    return (math.degrees(math.atan2(u, v_norte)) + 360.0) % 360.0


def nombre_rumbo(grados):
    pts = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSO", "SO", "OSO", "O", "ONO", "NO", "NNO"]
    return pts[int((grados + 11.25) % 360 // 22.5)]


# ==========================================
# 5. INTERFAZ
# ==========================================
st.set_page_config(page_title="Radar Nowcasting", layout="wide")
st.title("⛈️ Predictor de Impacto de Precipitación")

DEFAULTS = {
    "lat": 39.47, "lon": -0.37,
    "u_850": 12.0, "v_850": 5.0, "u_700": 15.0, "v_700": 8.0,
    "u_500": 20.0, "v_500": 12.0, "u_shr": 18.0, "v_shr": 10.0,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def _cb_viento():
    try:
        st.session_state.update(
            obtener_vientos_open_meteo(st.session_state["lat"], st.session_state["lon"]))
        st.session_state["msg_viento"] = ("ok", "Vientos actualizados (Open-Meteo, hora actual UTC).")
    except Exception as e:
        st.session_state["msg_viento"] = ("error", f"Error con Open-Meteo: {e}")


st.sidebar.header("1. Tu ubicación")
lat_destino = st.sidebar.number_input("Latitud", key="lat", format="%.4f")
lon_destino = st.sidebar.number_input("Longitud", key="lon", format="%.4f")

st.sidebar.header("2. Viento (m/s: u→este, v→norte)")
st.sidebar.button("🌐 Auto-completar viento (Open-Meteo)", type="primary", on_click=_cb_viento)
if "msg_viento" in st.session_state:
    tipo, texto = st.session_state["msg_viento"]
    (st.sidebar.success if tipo == "ok" else st.sidebar.error)(texto)

c1, c2 = st.sidebar.columns(2)
u_850 = c1.number_input("U 850hPa", key="u_850", format="%.2f")
v_850 = c2.number_input("V 850hPa", key="v_850", format="%.2f")
u_700 = c1.number_input("U 700hPa", key="u_700", format="%.2f")
v_700 = c2.number_input("V 700hPa", key="v_700", format="%.2f")
u_500 = c1.number_input("U 500hPa", key="u_500", format="%.2f")
v_500 = c2.number_input("V 500hPa", key="v_500", format="%.2f")
u_shr = c1.number_input("U shear", key="u_shr", format="%.2f")
v_shr = c2.number_input("V shear", key="v_shr", format="%.2f")
cizalladura = math.hypot(u_shr, v_shr)
st.sidebar.caption(f"|Cizalladura| ≈ {cizalladura:.1f} m/s (500 hPa − superficie)")
usar_bunkers = st.sidebar.checkbox("Desviación de Bunkers (supercélulas)", value=True)

st.sidebar.header("3. Proyección y ETA")
horizonte = st.sidebar.slider("Horizonte máximo (min)", 30, 180, 120, step=PASO_MIN)
umbral_eta = st.sidebar.slider("Umbral de lluvia para el ETA (dBZ)", 10, 50, 20)
radio_eta = st.sidebar.slider("Radio alrededor del punto (km)", 1, 10, 3)

st.write("### Fuente de datos del radar")
metodo = st.radio("Elige la fuente:", [
    "RainViewer (frames reales)",
    "Datos sintéticos de prueba",
    "Archivo NetCDF (.nc)",
])

zoom = 7
archivo = None
if metodo.startswith("RainViewer"):
    zoom = st.select_slider(
        "Alcance del mapa", options=[6, 7], value=7,
        format_func=lambda z: f"z{z}: ±{128 * km_por_pixel(lat_destino, z):.0f} km, "
                              f"{km_por_pixel(lat_destino, z):.1f} km/px")
elif metodo.endswith("(.nc)"):
    archivo = st.file_uploader("Sube el volumen (tiempo, y, x) en dBZ", type=["nc"])
    n1, n2, n3, n4, n5 = st.columns(5)
    var_name = n1.text_input("Variable", value="reflectivity")
    res_km = n2.number_input("Resolución (km/px)", value=1.0, min_value=0.1)
    dt_min = n3.number_input("Minutos entre frames", value=10.0, min_value=1.0)
    px_nc = n4.number_input("Columna de tu punto", value=0, min_value=0, step=1)
    py_nc = n5.number_input("Fila de tu punto", value=0, min_value=0, step=1)

if st.button("Ejecutar Nowcasting", type="primary"):
    try:
        with st.spinner("Obteniendo datos de radar..."):
            if metodo.startswith("RainViewer"):
                datos = cargar_rainviewer(lat_destino, lon_destino, zoom)
            elif metodo.startswith("Datos sint"):
                datos = datos_sinteticos()
            else:
                if archivo is None:
                    st.error("Sube un archivo .nc primero.")
                    st.stop()
                datos = cargar_netcdf(archivo, var_name, res_km, dt_min, px_nc, py_nc)
    except Exception as e:
        st.error(f"No se pudieron obtener los datos de radar: {e}")
        st.stop()

    with st.spinner("Calculando flujo óptico y proyección..."):
        campos, kmpp = datos["campos"], datos["kmpp"]
        dbz0 = campos[-1]
        u_r, v_r = flujo_medio(campos, datos["dt_s"], kmpp)

        u_g, v_g = viento_guia(u_850, v_850, u_700, v_700, u_500, v_500)
        if usar_bunkers:
            u_env, v_env = vector_bunkers(u_g, v_g, u_shr, v_shr)
        else:
            u_env, v_env = u_g, v_g

        frames, u_ini, v_ini = pronosticar(
            dbz0, u_r, v_r, u_env, -v_env, kmpp, horizonte)  # v_sur = -v_norte

        activo = dbz0 >= 25
        if activo.any():
            u_cel, v_cel = float(u_ini[activo].mean()), float(-v_ini[activo].mean())
        else:
            u_cel, v_cel = u_env, v_env

    st.session_state["res"] = {
        "datos": datos, "frames": frames, "horizonte": horizonte,
        "u_env": u_env, "v_env": v_env, "u_cel": u_cel, "v_cel": v_cel,
        "hay_eco": bool(activo.any()),
    }

# --- RESULTADOS (persisten al mover sliders) ---
res = st.session_state.get("res")
if res:
    datos, frames = res["datos"], res["frames"]
    kmpp, px, py, t_ref = datos["kmpp"], datos["px"], datos["py"], datos["t_ref"]

    st.success(f"Cálculo finalizado · {datos['origen']}")
    if t_ref:
        edad = (datetime.now(timezone.utc).timestamp() - t_ref) / 60.0
        st.caption(f"Último frame: {hora_local(t_ref)} (hace {edad:.0f} min)")

    t_eta, m_eta = calcular_eta(frames, kmpp, px, py, radio_eta, umbral_eta)
    st.write("### ⏱️ Impacto estimado (ETA)")
    if t_eta is None:
        st.markdown(f"> No se prevé lluvia ≥ {umbral_eta} dBZ sobre tu ubicación "
                    f"en los próximos {res['horizonte']} min.")
    elif t_eta == 0:
        st.markdown(f"> Ya hay lluvia de {m_eta:.0f} dBZ sobre tu ubicación.")
    else:
        if t_ref:
            desde_ahora = max(t_eta - edad, 0.0)
            st.markdown(f"> Llegada estimada en **{desde_ahora:.0f} min** "
                        f"(hacia las {hora_local(t_ref + t_eta * 60)}), "
                        f"intensidad ≈ {m_eta:.0f} dBZ.")
        else:
            st.markdown(f"> Llegada estimada en **{t_eta} min** desde el último frame, "
                        f"intensidad ≈ {m_eta:.0f} dBZ.")

    vel_kmh = math.hypot(res["u_cel"], res["v_cel"]) * 3.6
    rumbo = rumbo_hacia(res["u_cel"], res["v_cel"])
    if res["hay_eco"]:
        st.caption(f"Movimiento de las células: {vel_kmh:.0f} km/h hacia "
                   f"{nombre_rumbo(rumbo)} ({rumbo:.0f}°). Viento ambiente usado: "
                   f"{math.hypot(res['u_env'], res['v_env']) * 3.6:.0f} km/h.")
    else:
        st.caption("No hay ecos ≥ 25 dBZ en el dominio; se usa sólo el viento ambiente.")

    tau = st.slider("Mapa a +X minutos", 0, res["horizonte"], min(30, res["horizonte"]), step=PASO_MIN)
    col_a, col_b = st.columns(2)
    col_a.write("#### Radar actual")
    col_a.image(render(frames[0], kmpp, px, py))
    col_b.write(f"#### Proyección a +{tau} min")
    col_b.image(render(frames[tau], kmpp, px, py))
    st.caption("Escala fija 0–60 dBZ (azul ≈ 10–20, verde/amarillo ≈ 30–40, rojo ≥ 50). "
               "Círculo blanco = tu ubicación; anillos a 25/50/100 km. "
               "Fuera del dominio del mapa se asume que no hay lluvia.")
