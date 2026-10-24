# Conceptos de Machine Learning para este Proyecto

Guía rápida de los conceptos ML que usamos.

---

## 1. Clasificación vs Regresión

| Tipo              | Output             | Ejemplo                             |
| ----------------- | ------------------ | ----------------------------------- |
| **Clasificación** | Categoría discreta | Flor: setosa, versicolor, virginica |
| Regresión         | Número continuo    | Precio: $150,000                    |

Este proyecto es de **clasificación multiclase** (3 clases).

---

## 2. Train/Test Split

```mermaid
flowchart LR
    dataset["Dataset completo<br/>150 muestras"] --> train["Train set<br/>80% = 120 muestras<br/>Para entrenar el modelo"]
    dataset --> test["Test set<br/>20% = 30 muestras<br/>Para evaluar el modelo"]
```

**¿Por qué separar?**

- Evitar "hacer trampa" evaluando con datos que el modelo ya vio
- Simular cómo funcionará con datos nuevos

**random_state=42**: Semilla para reproducibilidad. Siempre obtienes la misma división.

---

## 3. Normalización (StandardScaler)

Transforma features para que tengan:

- Media = 0
- Desviación estándar = 1

```
Antes:  [5.1, 3.5, 1.4, 0.2]  (diferentes escalas)
Después: [-0.9, 1.0, -1.3, -1.3]  (misma escala)
```

**¿Por qué normalizar?**

- Algunos algoritmos son sensibles a la escala
- Mejora convergencia en gradient descent
- Random Forest no lo necesita estrictamente, pero es buena práctica

---

## 4. Validación Cruzada (Cross-Validation)

En lugar de un solo train/test split, hacemos varios:

```mermaid
block-beta
    columns 7
    f1["Fold 1"] v1["Val"] t11["Train"] t12["Train"] t13["Train"] t14["Train"] s1["acc 0.95"]
    f2["Fold 2"] t21["Train"] v2["Val"] t22["Train"] t23["Train"] t24["Train"] s2["acc 0.93"]
    f3["Fold 3"] t31["Train"] t32["Train"] v3["Val"] t33["Train"] t34["Train"] s3["acc 0.97"]
    f4["Fold 4"] t41["Train"] t42["Train"] t43["Train"] v4["Val"] t44["Train"] s4["acc 0.94"]
    f5["Fold 5"] t51["Train"] t52["Train"] t53["Train"] t54["Train"] v5["Val"] s5["acc 0.96"]
    classDef val fill:#ffe599,stroke:#d6b656
    classDef trn fill:#dae8fc,stroke:#6c8ebf
    class v1,v2,v3,v4,v5 val
    class t11,t12,t13,t14,t21,t22,t23,t24,t31,t32,t33,t34,t41,t42,t43,t44,t51,t52,t53,t54 trn
```

Promedio: 0.95 ± 0.02 (valores ilustrativos). En cada fold, una parte distinta de los datos se usa como validación y el resto como entrenamiento.

**¿Por qué?**

- Más robusto que un solo split
- Detecta si el modelo es estable
- Usa todos los datos para entrenamiento y evaluación

---

## 5. Métricas de Evaluación

### Accuracy

```
Accuracy = Predicciones correctas / Total predicciones
         = 28/30 = 0.93
```

### Precision, Recall, F1

Para cada clase:

```mermaid
block-beta
    columns 3
    esquina[" "] pp["Predicho Pos"] pn["Predicho Neg"]
    ap["Actual Pos"] tp["TP"] fn["FN"]
    an["Actual Neg"] fp["FP"] tn["TN"]
    classDef ok fill:#d5e8d4,stroke:#82b366
    classDef err fill:#f8cecc,stroke:#b85450
    classDef header fill:#f5f5f5,stroke:#999999
    class tp,tn ok
    class fn,fp err
    class esquina,pp,pn,ap,an header
```

```
Precision = TP / (TP + FP)  → "De los que predije positivos, ¿cuántos lo eran?"
Recall    = TP / (TP + FN)  → "De los positivos reales, ¿cuántos encontré?"
F1        = 2 * (P * R) / (P + R)  → Balance entre precision y recall
```

### Matriz de Confusión

```mermaid
block-beta
    columns 5
    esquina[" "] p0["Pred 0"] p1["Pred 1"] p2["Pred 2"] nota["Aciertos"]
    r0["Real 0"] c00["10"] c01["0"] c02["0"] n0["Setosa: 10/10"]
    r1["Real 1"] c10["0"] c11["9"] c12["1"] n1["Versicolor: 9/10"]
    r2["Real 2"] c20["0"] c21["2"] c22["8"] n2["Virginica: 8/10"]
    classDef ok fill:#d5e8d4,stroke:#82b366
    classDef err fill:#f8cecc,stroke:#b85450
    classDef header fill:#f5f5f5,stroke:#999999
    class c00,c11,c22 ok
    class c12,c21 err
    class esquina,p0,p1,p2,nota,r0,r1,r2,n0,n1,n2 header
```

---

## 6. Probabilidades vs Predicción

```python
# Predicción: la clase más probable
model.predict([[5.1, 3.5, 1.4, 0.2]])
# Output: [0]  (clase 0 = setosa)

# Probabilidades: confianza por clase
model.predict_proba([[5.1, 3.5, 1.4, 0.2]])
# Output: [[0.95, 0.03, 0.02]]
#          setosa versicolor virginica
```

**¿Por qué retornar probabilidades?**

- Permite al cliente decidir umbral de confianza
- Útil para casos donde la predicción es incierta
- Mejor para debugging y análisis

---

## 7. Overfitting vs Underfitting

```mermaid
%%{init: {"themeVariables": {"xyChart": {"plotColorPalette": "#d62728, #1f77b4"}}}}%%
xychart-beta
    title "Error vs complejidad del modelo (ilustrativo)"
    x-axis "Complejidad del modelo" ["Muy baja", "Baja", "Media", "Alta", "Muy alta"]
    y-axis "Error" 0 --> 0.6
    line [0.50, 0.30, 0.18, 0.28, 0.45]
    line [0.48, 0.26, 0.12, 0.05, 0.01]
```

Curva roja: error en datos nuevos (validación). Curva azul: error en entrenamiento. A la izquierda hay underfitting (modelo simple); a la derecha, overfitting (modelo complejo).

**Underfitting**: Modelo muy simple, no captura patrones
**Overfitting**: Modelo memoriza datos de entrenamiento, no generaliza

**Random Forest evita overfitting** porque:

- Promedia muchos árboles (reduce varianza)
- Cada árbol ve subset aleatorio de datos y features

---

## 8. Hiperparámetros de Random Forest

| Parámetro           | Qué hace                      | Valor típico      |
| ------------------- | ----------------------------- | ----------------- |
| `n_estimators`      | Número de árboles             | 100               |
| `max_depth`         | Profundidad máxima            | None (sin límite) |
| `min_samples_split` | Mínimo para dividir nodo      | 2                 |
| `random_state`      | Semilla para reproducibilidad | 42                |

```python
RandomForestClassifier(
    n_estimators=100,      # Más árboles = más robusto, más lento
    max_depth=None,        # None = árboles crecen hasta ser puros
    min_samples_split=2,   # Mínimo 2 muestras para dividir
    random_state=42        # Reproducibilidad
)
```

---

## Recursos para Profundizar

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Random Forest Explained](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
- [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
