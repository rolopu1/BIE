import pandas as pd
import csv
import plotly.graph_objects as go

path = "C:\\BSICoS\\Maxim\\Ensayos\\Ensayo_Oct27_1157_AddFinalBC34"
Data = pd.read_csv(path+"\\MAX103_PPG.csv")

print(Data.head())


CANAL = "Verde"  # También puedes elegir "Rojo" o "Infrarrojo".

ppg = Data[CANAL].to_numpy()
tiempo = Data["Tiempo"].to_numpy()

# 3. DETECCIONES: aquí añadiremos el algoritmo en una próxima actividad.
# Cada número es la posición de una muestra en las listas, empezando en 0.
# Ejemplo para probar marcadores: detecciones = [100, 200, 300]
# Estos números son solo un ejemplo: no son latidos detectados.
detecciones = []


# 4. GRÁFICA INTERACTIVA
figura = go.Figure()
figura.add_trace(go.Scatter(
    x=tiempo, y=ppg, mode="lines", name=f"PPG {CANAL}",
    line=dict(color="forestgreen"),
))

figura.update_layout(
    title=f"Señal PPG - canal {CANAL}",
    xaxis_title="Tiempo (s)", yaxis_title="Amplitud (cuentas del sensor)",
    template="plotly_white", hovermode="x unified",
)
# Al principio se muestran 10 segundos. El resto sigue disponible al alejar.
figura.update_xaxes(range=[0, 10])
figura.show()