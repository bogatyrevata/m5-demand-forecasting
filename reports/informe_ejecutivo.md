# Informe ejecutivo – Predicción de demanda M5 (Walmart, California)

**Autora:** [Taisia Bogatyreva] · **Fecha:** [...]

## 1. Resumen de resultados
Se construyó un pipeline de pronóstico de ventas a 28 días por producto y tienda para California (4 tiendas, 12 196 series, `d_1500 … d_1913`), con validación en los últimos 28 días conocidos (`d_1886 … d_1913`).

| Modelo | RMSE | MAE |
|---|---|---|
| Ingenuo estacional | 2,8315 | 1,3244 |
| Media móvil 28 días (mejor baseline) | 2,2604 | 1,1070 |
| Predicción cero | 3,8650 | 1,4764 |
| **LightGBM** | **2,1613** | 1,1097 |

- LightGBM reduce el RMSE un **4,4 %** frente al mejor baseline; en MAE el resultado es equivalente (−0,2 %).
- La mejora se concentra en `FOODS` (RMSE −5,8 %, 63,7 % de las unidades) y crece con el horizonte: −0,8 % en la semana 1 y −8,4 % en la semana 4.
- El modelo no presenta sesgo sistemático (predicción media 1,4741 frente a venta real 1,4764).
- Las variables más importantes son las medias móviles de 7 y 28 días (79,5 % de la ganancia); los precios aportan ≈ 6,8 %; el calendario individual aporta poco.
- El 57 % de los días-serie tiene venta 0 (demanda intermitente), lo que limita las ganancias posibles.

## 2. Impacto de los eventos de calendario
- **Cierre de tiendas:** Christmas (−100 %).
- **Caídas fuertes:** Thanksgiving (−34,7 %), Mother's Day (−22,8 %), Independence Day (−19,6 %), New Year (−18,7 %), Father's Day (−13,4 %), Super Bowl (−12,8 %).
- **Aumentos:** Labor Day (+31,5 %), Presidents Day (+19,3 %), Columbus Day (+14,6 %), Martin Luther King Day (+11,7 %), todos fines de semana largos.
- Los días SNAP (días 1-10 de cada mes) venden un 5,8 % más, pero el efecto no se puede separar del efecto "inicio de mes".
- Sábado y domingo venden un 40-45 % más que martes-jueves.
- Cada evento aparece solo 1-2 veces en la ventana: los porcentajes son indicativos, no concluyentes.

## 3. Recomendaciones para la cadena de suministro
1. **Usar el modelo como base de planificación en horizontes de 2-4 semanas**, donde su ventaja sobre la media móvil es mayor (RMSE −5 % a −8 %). Para la primera semana, la media móvil reciente es una alternativa igual de válida.
2. **Concentrar el esfuerzo en `FOODS`:** 63,7 % de las unidades y donde el modelo mejora más.
3. **Calendario operativo de eventos:** ajustar pedidos a la baja en Thanksgiving, Día de la Madre, Día de la Independencia y Año Nuevo; planificar cero reposición el 25 de diciembre y reforzar el inventario antes de los fines de semana largos.
4. **Reforzar el stock en fines de semana y en los primeros 10 días del mes**, cuando la demanda es mayor.
5. **Sin sesgo sistemático**, pero el costo asimétrico (faltante vs. exceso) no fue evaluado; antes de fijar inventarios de seguridad conviene medirlo con datos de costo reales.
6. **Series de baja rotación:** para las series con más de 80 % de días en cero (21,4 %), el valor del modelo es bajo; conviene gestionarlas con reglas simples de reposición.

*Nota:* las recomendaciones 3 y 4 se basan en patrones descriptivos con pocas observaciones por evento y deben validarse con más historia.

## 4. Limitaciones y próximos pasos
- No se calculó WRMSSE (restricción de tiempo); se reportan RMSE y MAE, que no ponderan por valor monetario.
- Un único corte de validación; solo CA y 414 días de historia; sin búsqueda de hiperparámetros.
- **Próximos pasos:** (1) estrategia recursiva o combinación con la media reciente para los primeros días del horizonte; (2) expanding window con varios cortes; (3) calcular WRMSSE; (4) ampliar a TX y WI y a más historia; (5) probar un objetivo orientado a la mediana o cuantiles para mejorar el MAE.