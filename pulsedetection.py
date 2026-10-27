"""Traducción Python de pulsedetection.m de BSICoS/biosigmat (GPL-3.0).

Origen: https://github.com/BSICoS/biosigmat
Revisión: f872d957129aff78d1835b760e91e7a811f83ec5.
Incluye adaptiveThreshold, localmax y refinamiento QLS de tres puntos.
Entrada: DERIVADA FILTRADA de la PPG; salida: tiempos de máxima pendiente
(en segundos desde t=0) y umbral. No detecta máximos sistólicos de PPG.

Conserva bloques de 10000 muestras, extensión de 5 s y parámetros originales.
Corrige los límites de bloques por la conversión de índices desde MATLAB.
Redondea la extensión para fs no entero y limita su tamaño a un bloque.
Protege casos sin datos válidos y duraciones que redondean a cero.
No se ha comprobado igualdad numérica ejecutando MATLAB.
Requiere numpy y scipy.
"""
import numpy as np
from scipy.signal import find_peaks

__all__ = ['pulsedetection']


def _round(value):
    # MATLAB redondea las mitades positivas hacia arriba.
    return int(np.floor(value + 0.5))


def _positive(value, name):
    a = np.asarray(value)
    if a.ndim != 0 or a.dtype.kind not in 'iuf' or not np.isfinite(a) or a <= 0:
        raise ValueError(f'{name} debe ser un número finito mayor que cero.')
    return float(a)


def pulsedetection(dppg, fs, *, method='adaptive', adaptive_alpha_amp=0.2,
                   adaptive_refract_period=0.15, adaptive_tau_rr=1.0):
    """Devuelve nD, threshold (arrays 1D).

    dppg: derivada filtrada de PPG (vector); admite NaN, no infinito.
    fs: frecuencia de muestreo en Hz.
    method: 'adaptive' (único método del original).
    adaptive_alpha_amp: mínimo del umbral como fracción de amplitud (0.2).
    adaptive_refract_period: período refractario en segundos (0.15).
    adaptive_tau_rr: duración de caída como fracción del RR estimado (1).

    nD contiene TIEMPOS refinados en segundos, no índices.
    threshold tiene la misma longitud y unidades que dppg.
    Para dibujar con un eje de tiempo no iniciado en cero, sumar tiempo[0].
    Índices aproximados: np.floor(nD * fs + 0.5).astype(int).
    """
    a = np.asarray(dppg)
    if (a.size == 0 or a.dtype.kind not in 'iuf' or a.ndim > 2
            or a.ndim == 0 or (a.ndim == 2 and 1 not in a.shape)):
        raise ValueError('dppg debe ser un vector numérico real no vacío.')
    x = a.astype(float).reshape(-1)
    if np.isinf(x).any():
        raise ValueError('dppg puede contener NaN, pero no infinitos.')
    fs = _positive(fs, 'fs')
    alpha = _positive(adaptive_alpha_amp, 'adaptive_alpha_amp')
    ref = _positive(adaptive_refract_period, 'adaptive_refract_period')
    tau = _positive(adaptive_tau_rr, 'adaptive_tau_rr')
    if not isinstance(method, str) or method.lower() != 'adaptive':
        raise ValueError("El único método disponible es 'adaptive'.")
    size = 10000
    extra = min(size, _round(5 * fs))
    threshold = np.full(x.size, np.nan)
    detected = []
    nblocks = (x.size + size - 1) // size
    # slicesignal(..., Uselast=true) rellena el último bloque con NaN.
    padded = np.pad(x, (0, nblocks * size - x.size), constant_values=np.nan) if nblocks > 1 else x
    for start in range(0, padded.size, size):
        left = max(0, start - extra)
        right = min(padded.size, start + size + extra)
        peaks, level = _adaptive(padded[left:right], fs, alpha, ref, tau)
        offset = start - left
        count = min(size, x.size - start)
        threshold[start:start + count] = level[offset:offset + count]
        # Intervalos semiabiertos: sin muestras repetidas en los límites.
        keep = (peaks >= offset) & (peaks < offset + count)
        detected.extend((peaks[keep] + left).tolist())
    indices = np.unique(np.asarray(detected, dtype=int))
    return _refine(x, indices, fs), threshold


def _adaptive(x, fs, alpha, refractory, tau):
    n = x.size
    valid = np.flatnonzero(~np.isnan(x))
    if not valid.size:
        return np.empty(0, dtype=int), np.full(n, np.nan)
    initial = x[valid[0]:min(n, valid[0] + _round(10 * fs) + 1)]
    positive = initial[initial >= 0]
    if not positive.size:
        return np.empty(0, dtype=int), np.full(n, np.nan)
    amplitude = 3 * np.mean(positive)
    rr = max(1, _round(0.75 * fs))
    ref = max(1, _round(refractory * fs))
    level = amplitude - amplitude * (1 - alpha) / rr * np.minimum(np.arange(n), rr)
    peaks = []
    cursor = 0
    while cursor < n:
        up = np.flatnonzero(x[cursor:] > level[cursor:])
        if not up.size:
            break
        up = cursor + int(up[0])
        down = np.flatnonzero(x[up:] < level[up:])
        if not down.size:
            break
        down = up + int(down[0])
        region = x[up:down + 1]
        candidates, props = find_peaks(region, plateau_size=True)
        if not candidates.size:
            cursor = down + 1
            continue
        # islocalmax FlatSelection='center': centro derecho de mesetas pares.
        candidates = (props['left_edges'] + props['right_edges'] + 1) // 2
        peak = up + int(candidates[np.argmax(region[candidates])])
        current = np.max(region)
        if len(peaks) > 4:
            current = np.median(np.r_[x[peaks[-4:]], current])
        peaks.append(peak)
        if len(peaks) >= 4:
            rr = max(1, _round(np.median(np.diff(peaks[-4:]))))
        elif len(peaks) >= 2:
            rr = max(1, _round(np.mean(np.diff(peaks))))
        ref_end = min(peak + ref, n - 1)
        cursor = max(ref_end, cursor + 1)
        level[peak:ref_end + 1] = current
        minimum = alpha * current
        if len(peaks) >= 4:
            estimate = np.median(x[peaks[-4:-1]])
            if current >= 2 * estimate:
                minimum = alpha * estimate
                current = estimate
        duration = max(1, _round(tau * rr))
        end = min(ref_end + duration, n - 1)
        level[ref_end:end + 1] = current - (current - minimum) / duration * np.arange(end - ref_end + 1)
        level[end + 1:] = minimum
    return np.unique(np.asarray(peaks, dtype=int)), level


def _refine(x, indices, fs):
    # refinepeaks de MATLAB usa QLS sin transformación como opción por defecto.
    result = []
    for p in indices:
        if p == 0 or p == x.size - 1:
            continue
        left, center, right = x[p - 1:p + 2]
        curvature = left - 2 * center + right
        if not np.isfinite([left, center, right]).all() or curvature >= 0:
            continue
        shift = 0.5 * (left - right) / curvature
        result.append((p + shift) / fs)
    return np.unique(np.asarray(result, dtype=float))
