<!-- =======================
      BANNER DEL PROYECTO
======================= -->

<p align="center">
  <img src="https://img.shields.io/badge/Team%20Challenge-Machine%20Learning%20Toolbox-4B9CD3?style=for-the-badge&logo=python&logoColor=white" />
</p>

<h1 align="center">🧰 Team 3_Toolbox</h1>
<h3 align="center">Funciones reutilizables para análisis exploratorio y selección de variables</h3>

<p align="center">
  <img src="https://img.shields.io/badge/EDA-Exploratory%20Data%20Analysis-0A84FF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Regression-Feature%20Selection-34C759?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Visualization-Seaborn%20%7C%20Matplotlib-FF9500?style=for-the-badge" />
</p>

<hr>

# 📝 Descripción del Proyecto

Este repositorio contiene un **Toolbox de funciones en Python** diseñadas para automatizar tareas frecuentes en Machine Learning:

- Clasificación automática de variables (categóricas, binarias y numéricas)
- Selección de variables relevantes para regresión basadas en correlación
- Visualizaciones automáticas con *pairplots* filtradas por relevancia
- Funciones 100% reutilizables para cualquier dataset tabular

El trabajo forma parte del **Team Challenge del Bootcamp de Data Science**.

---

## 🎯 **Objetivo del Proyecto**
Crear un conjunto de funciones que permita:

- Identificar y clasificar el tipo de cada variable
- Seleccionar automáticamente las mejores columnas predictoras (numéricas y categóricas)
- Visualizar relaciones clave entre variables y target
- Facilitar un análisis estadístico claro y reproducible

---

## 🧩 **¿Qué contiene este Toolbox?**
- Funciones para **EDA automatizado**
- Selección de *features* numéricas por correlación + significación estadística
- Selección de *features* categóricas por test (según corresponda)
- Visualizaciones automáticas (pairplots)
- Tests completos incluidos en `Test.ipynb`  
- Compatible con cualquier dataset orientado a regresión

---
  
# 📁 Estructura del Repositorio
CHALLENGE-TOOLBOX/  
│
├── data/  
│ 
│ └── dataset_viajes_jun23.csv  
│
├── toolbox_ML.py # Archivo principal con TODAS las funciones  
├── Team_Challenge_ToolBox.ipynb # Notebook con pruebas del Toolbox  
├── Test.ipynb # Tests de validación de funciones  
└── README.md # Este archivo  


---

# 🧠 Funciones Incluidas

## 1️⃣ `describe_df(df)`
Genera una tabla con:
- Tipo de dato  
- Nulos  
- Valores únicos  
- Cardinalidad (%)  

Es usada internamente por otras funciones.

---

## 2️⃣ `tipifica_variables(df, umbral_categoria, umbral_continua)`
Clasifica automáticamente las columnas del dataframe en:

- **Binaria**  
- **Categórica**  
- **Numérica Discreta**
- **Numérica Continua**

Basada en:
- Número de valores únicos  
- Porcentaje de cardinalidad  
- Umbrales ajustables por el usuario  

---

## 3️⃣ `get_features_num_regression(df, target_col, umbral_corr, pvalue=None)`
Selecciona columnas numéricas relevantes para regresión.

✔ Mantiene solo columnas con correlación absoluta con el *target*  
✔ Si se indica `pvalue`, filtra también por significación estadística  
✔ Devuelve lista de columnas recomendadas  

---

## 4️⃣ `plot_features_num_regression(df, target_col, columns, umbral_corr, pvalue)`
Genera automáticamente:

- Pairplots de variables relevantes  
- Gráficos limpios y filtrados  
- Máximo 5 columnas por gráfico  
- Siempre aparece el target  

También devuelve la lista final de columnas graficadas.

---

## 5️⃣ `get_features_cat_regression(df, target_col, pvalue=0.05)`
Selecciona columnas **categóricas** relevantes para un modelo de regresión con target numérico.

🔍 Para cada variable categórica:
- Aplica el **test estadístico adecuado** según el tipo de variable (binaria / con más categorías).  
- Evalúa si la relación con el target es **estadísticamente significativa** (según `pvalue`).

Solo devuelve aquellas columnas categóricas cuyo test de relación con el target es significativo.

✔ Devuelve una **lista de columnas categóricas** recomendadas.

---

## 6️⃣ `plot_features_cat_regression(df, target_col, columns=[], pvalue=0.05, with_individual_plot=False)`
Genera gráficos para analizar la relación entre el **target numérico** y las **variables categóricas**.

- Si `columns` está vacío → utiliza automáticamente todas las variables categóricas del dataframe (o las devueltas por `get_features_cat_regression`).  
- Dibuja **histogramas/boxplots agrupados** del target para cada categoría.  
- Si `with_individual_plot = True` → un gráfico por variable.  
- Si `with_individual_plot = False` → puede agrupar varias variables en menos figuras.  

Devuelve la lista final de columnas categóricas que se han graficado.


# 🧪 Tests Incluidos

En el archivo `Test.ipynb` se prueban TODAS las funciones con el siguiente datasets:

### ✔ Dataset 2: *dataset_viajes_jun23.csv*
- Tipificación variada (categorías, numéricas discretas, continuas)  
- Evaluación del comportamiento con cardinalidades altas/bajas  
- Validación de correlaciones lógicas  

---
✨ Requisitos   
Python  
Pandas   
Numpy   
Seaborn  
matplotlib  
scipy  

----
Autores:

Brenda Oyola  
Diana Hoyos  
Elena Acosta  
Lucía Fuentes  











