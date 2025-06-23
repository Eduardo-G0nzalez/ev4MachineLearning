# ev4MachineLearning

En este caso la **hipótesis** que desarrollamos fue: *"Con base en el equipamiento inicial del jugador, el tipo de arma principal utilizada y sus estadísticas de combate en la ronda, es posible predecir si el jugador sobrevivirá o no la ronda”.*

**Los resultados obtenidos fueron:**
- **kNN (k=20):** Precisión del 66.9%.
- **Árbol de Decisión:** Precisión del 63%, mejor para predecir no supervivencia.
- **Random Forest:** Precisión del 68%, sólido y equilibrado.
- **Regresión Logística:** Precisión del 68%, con buena interpretabilidad.
- **SVM:** Precisión del 68%, rendimiento consistente.

**Conclusiones generales:**
- Los modelos evaluados muestran un rendimiento consistente, con precisiones cercanas al 68%. Random Forest, Regresión Logística y SVM se destacan como los más confiables. Aunque el Árbol de Decisión tuvo menor precisión, permite una visualización clara de las decisiones. En general, es posible predecir la supervivencia del jugador con un nivel razonable de acierto utilizando solo sus estadísticas de combate y equipamiento inicial.
