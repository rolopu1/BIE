"""Práctica BIE: representar PPG y detectar máximos de pendiente con Plotly."""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, sosfiltfilt
from pulsedetection import pulsedetection

# CONFIGURACIÓN
archivo = Path(r'C:\BSICoS\Maxim\Ensayos\Ensayo_Oct27_1157_AddFinalBC34\MAX103_PPG.csv')
# Este ensayo tiene un salto grande al inicio. Cambiar a 0 para otros registros.
recorte_inicio_s = 1
recorte_final_s = 0
frecuencia_corte_hz = 8.0

# 1. LEER LOS DATOS
data = pd.read_csv(archivo)
tiempo = data['Tiempo_s'].to_numpy(dtype=float)
ppgGreen = -data['Verde'].to_numpy(dtype=float)
ppgRed = data['Rojo'].to_numpy(dtype=float)
ppgIR = data['Infrarrojo'].to_numpy(dtype=float)
if len(tiempo) < 10 or not np.isfinite(tiempo).all():
    raise ValueError('El registro no tiene suficientes tiempos válidos.')
dt = np.diff(tiempo)
if np.any(dt <= 0):
    raise ValueError('Los tiempos deben ser estrictamente crecientes.')
fs = 1.0 / np.median(np.diff(tiempo))
if np.max(np.abs(dt - np.median(dt))) > 0.05 * np.median(dt):
    raise ValueError('El muestreo no es uniforme; revisar los tiempos antes de filtrar.')

# 2. REPRESENTAR LOS TRES CANALES ORIGINALES
figura = make_subplots(rows=3, cols=1, shared_xaxes=True,
                      subplot_titles=('PPG verde', 'PPG roja', 'PPG infrarroja'))
for fila, (ppg, nombre, color) in enumerate([
        (ppgGreen, 'Verde', 'forestgreen'),
        (ppgRed, 'Rojo', 'red'),
        (ppgIR, 'Infrarrojo', 'darkorange')], start=1):
    figura.add_trace(go.Scatter(x=tiempo, y=ppg, name=nombre,
                               line=dict(color=color)), row=fila, col=1)
    figura.update_yaxes(title_text='PPG (cuentas)', row=fila, col=1)
figura.update_xaxes(title_text='Tiempo (s)', row=3, col=1)
figura.update_layout(title='Señales PPG originales', template='plotly_white',
                     hovermode='x unified', height=750)

# 3. PREPARAR LA DERIVADA FILTRADA
inicio = int(np.ceil(recorte_inicio_s * fs))
fin = len(tiempo) - int(np.ceil(recorte_final_s * fs))
if fin - inicio <= max(30, int(fs)):
    raise ValueError('El recorte deja una señal demasiado corta.')
t = tiempo[inicio:fin]
x = ppgGreen[inicio:fin]
if not np.isfinite(x).all():
    raise ValueError('La PPG contiene datos no válidos; revisar el registro.')
# Butterworth + derivada: preprocesado para esta práctica, distinto del FIR LPD de MATLAB.
sos = butter(4, frecuencia_corte_hz, fs=fs, output='sos')
ppg_filtrada = sosfiltfilt(sos, x - np.mean(x))
dppg = np.gradient(ppg_filtrada, 1 / fs)

# 4. DETECTAR PULSOS (nD contiene segundos, no amplitudes ni índices)
nD, umbral = pulsedetection(dppg, fs, adaptive_alpha_amp=0.1, adaptive_refract_period=0.15, adaptive_tau_rr=0.5)
tiempo_detecciones = t[0] + nD
amplitud_ppg = np.interp(tiempo_detecciones, t, ppg_filtrada)
amplitud_derivada = np.interp(tiempo_detecciones, t, dppg)

# 5. REPRESENTAR DETECCIONES Y UMBRAL
figura2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=('PPG filtrada y puntos de máxima pendiente',
                                       'Derivada filtrada y umbral adaptativo'))
figura2.add_trace(go.Scatter(x=t, y=ppg_filtrada, name='PPG filtrada',
                             line=dict(color='forestgreen')), row=1, col=1)
figura2.add_trace(go.Scatter(x=tiempo_detecciones, y=amplitud_ppg,
                             mode='markers', name='Máxima pendiente en PPG',
                             marker=dict(color='red', symbol='x', size=9)), row=1, col=1)
figura2.add_trace(go.Scatter(x=t, y=dppg, name='Derivada filtrada'), row=2, col=1)
figura2.add_trace(go.Scatter(x=t, y=umbral, name='Umbral',
                             line=dict(color='black', dash='dash')), row=2, col=1)
figura2.add_trace(go.Scatter(x=tiempo_detecciones, y=amplitud_derivada,
                             mode='markers', name='Pulsos detectados',
                             marker=dict(color='red', symbol='x', size=9)), row=2, col=1)
figura2.update_yaxes(title_text='PPG sin media (cuentas)', row=1, col=1)
figura2.update_yaxes(title_text='Derivada (cuentas/s)', row=2, col=1)
figura2.update_xaxes(title_text='Tiempo (s)', row=2, col=1)
figura2.update_layout(title='Detección de pulsos — canal verde',
                     template='plotly_white', hovermode='x unified', height=750)
figura2.update_xaxes(range=[float(t[0]), min(float(t[-1]), float(t[0] + 10))])
print(f'Frecuencia de muestreo estimada: {fs:.3f} Hz')
print(f'Pulsos detectados: {len(nD)}')
# Estos puntos marcan la máxima pendiente, no el máximo de amplitud de cada pulso.
figura.show()
figura2.show()
