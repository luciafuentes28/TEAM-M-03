
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns   
from statsmodels.stats.proportion import proportions_ztest


pd.options.mode.copy_on_write = True

def describe_df(df):
    '''
    Descripción: Para cada columna del dataframe, devuelve el tipo de variable, el
    porcentaje de nulos, los valores únicos y porcentaje de cardinalidad.

    Argumentos:
    df (dataframe)

    Retorna:
    dataframe: dataframe con las columnas del dataframe original, y con las filas
    de los tipos de variables, el tanto por ciento de nulos, los valores únicos
    y el porcentaje de cardinalidad.
    '''

    # Data_type
    data_type = df.dtypes

    # Nulos
    nulos = df.isna().sum()
    porcentaje_nulos = (nulos / len(df) * 100).round(2)

    # Valores únicos
    unicos = df.nunique()

    # Cardinalidad
    card = (unicos / len(df) * 100).round(2)

    # Creación tabla
    tabla = pd.DataFrame({
        "DATA_TYPE": data_type,
        "MISSINGS (%)": porcentaje_nulos,
        "UNIQUE_VALUES": unicos,
        "CARDIN (%)": card
    },
    index = pd.Index(df.columns, name = "COL_N"))

    return tabla.T # devuelve la traspuesta
 

def tipifica_variables(df, umbral_categoria, umbral_continua):
    '''
    Descripción:
    Clasifica las variables del dataframe df en Binaria, Categórica, Numérica continua
    o Numérica Discreta, según el siguiente criterio:
    - Si la cardinalidad es 2, asignara "Binaria"
    - Si la cardinalidad es menor que `umbral_categoria` asignara "Categórica"
    - Si la cardinalidad es mayor o igual que `umbral_categoria`, entonces entra en juego el tercer argumento:
        * Si además el porcentaje de cardinalidad es superior o igual a `umbral_continua`, asigna "Numerica Continua"
        * En caso contrario, asigna "Numerica Discreta"

    Argumentos:
    df (dataframe)
    umbral_categoria (entero): para determinar si una variable es categórica
    umbral_continua (float): para determinar si una variable numérica es continua o discreta

    Retorna:
    dataframe: dataframe con dos columnas "nombre_variable" y "tipo_sugerido".
    '''

    # Creo una lista vacía donde guardar luego los resultados
    resultados = []

    df_descripcion = describe_df(df).T # aprovecho la función describe_df que ya calcula la cardinalidad y hacemos la transpuesta, para que los nombres de
                                     # las variables sean las filas y no las columnas
    
    
    for col in df.columns:
        # Guardo los valores únicos y las cardinalidades en porcentajes de cada columna de df
        card = df_descripcion["UNIQUE_VALUES"][col]
        card_pct = df_descripcion["CARDIN (%)"][col]

        if card == 2:
            tipo = "Binaria"
        elif card < umbral_categoria:
            tipo = "Categórica"
        else:
            if card_pct >= umbral_continua:
                tipo = "Numérica Continua"
            else:
                tipo = "Numérica Discreta"

        resultados.append({"nombre_variable": col, "tipo_sugerido": tipo})

    return pd.DataFrame(resultados)


def get_features_num_regression(df, target_col, umbral_corr, pvalue = None):
    '''
    Descripción:
    Selecciona columnas numéricas que tengan una relación fuerte con la variable objetivo. 
    Solo se eligen las que superan el umbral de correlación y opcionalmente las que tienen un p-valor significativo.

    Argumentos: 
    df (dataframe)
    target_col (columna): columna de df, que debería ser el target de un hipotético modelo de regresión. Es una variable numérica con alta cardinalidad.
    umbral_corr (float): debe estar entre 0 y 1.
    pvalue (float, por defecto None): 

    Retorna:
    Lista con los nombres de las columnas numéricas seleccionadas que cumplen los criterios de correlación y, si aplica, de p-valor.
    En caso de error en los parámetros, retorna None.

    '''
    # 1. Comprobaciones básicas

    # Comprobamos que target_col exista en el dataframe  
    if target_col not in df.columns:
        print(f"ERROR: la columna '{target_col}' no existe en el DataFrame.")
        return None

    # Comprobamos que target_col es numérica
    if df[target_col].dtype not in ["int64", "float64"]:
        print("ERROR: 'target_col' debe ser una columna numérica.")
        return None
    
    # Comprobamos que umbral_corr está entre 0 y 1
    if not isinstance(umbral_corr, (int, float)) or not (0 <= umbral_corr <= 1):
        print("ERROR: 'umbral_corr' debe ser un número entre 0 y 1.")
        return None
    
    # Comprobamos pvalue si no es None
    if pvalue is not None:
        if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
            print("ERROR: 'pvalue' debe ser un número entre 0 y 1 o None.")
            return None

     # 2. Seleccionar columnas numéricas 
    
    columnas_numericas = []

    for col in df.columns:
        # añadimos solo columnas numéricas y que no sean el target
        if df[col].dtype in ["int64", "float64"] and col != target_col:
            columnas_numericas.append(col)

     # Si no hay columnas numéricas, devolvemos lista vacía
    if len(columnas_numericas) == 0:
        print("No hay columnas numéricas aparte del target.")
        return []

    columnas_seleccionadas = []
        
    # 3. Calcular correlaciones y p-valores
   
    for col in columnas_numericas:

        # Seleccionamos target + columna actual y quitamos filas con nulos
        datos = df[[target_col, col]].dropna()

        # evitar columnas con muy pocos datos
        if len(datos) < 3:
            continue
        # calculamos la correlación entre el target y una variable numérica,
        try:    
            # Cálculo de correlación y p-valor
            # stats.pearsonr(x, y) devuelve:
            #   - correlación entre x e y
            #   - p-valor del test de hipótesis
            corr, p_val = stats.pearsonr(datos[target_col], datos[col])
            
        except:
            continue

    # 4. Filtro por correlación
     
        if abs(corr) <= umbral_corr:
            continue

    # 5. Filtro por p-valor

        if pvalue is not None:
            if p_val > pvalue:
                continue

        columnas_seleccionadas.append(col)


    return columnas_seleccionadas



def plot_features_num_regression(df, target_col, columns, umbral_corr, pvalue):
    '''
    Descripción: 
    Selecciona las columnas numéricas del dataframe cuya relación con la variable objetivo supera un umbral indicado. Opcionalmente, 
    también filtra por significación estadística si se proporciona un valor p.

    Argumentos:
        df (DataFrame): 
            Conjunto de datos que contiene la variable objetivo y las 
            variables numéricas a analizar.
        target_col (str): 
            Nombre de la columna objetivo. Debe ser numérica continua 
            o discreta con alta cardinalidad.
        umbral_corr (float): 
            Valor entre 0 y 1 que define la relación mínima exigida para 
            considerar una variable relevante.
        pvalue (float o None): 
            Umbral de significación estadística. Si es None, no se aplica 
            este filtro adicional.

    Retorna:
        list: 
            Lista de nombres de columnas numéricas que cumplen los criterios 
            establecidos. Devuelve None si los argumentos no son válidos.
    '''
            

     #  COMPROBACIONES BÁSICAS
   
    # Comprobamos el target en el DataFrame
    if target_col not in df.columns:
        print("ERROR: la columna objetivo no existe.")
        return None

    # Comprobamos si el El target es numérico
    if df[target_col].dtype not in ["int64", "float64"]:
        print("ERROR: el target debe ser numérico.")
        return None

    # Comprobamos el umbral
    if not (0 <= umbral_corr <= 1):
        print("ERROR: umbral_corr debe estar entre 0 y 1.")
        return None

    # Comprobamos si el pvalue es válido
    if pvalue is not None:
        if not (0 < pvalue < 1):
            print("ERROR: pvalue debe ser un número entre 0 y 1 o None.")
            return None

    #2. Usaremos la columnas numéricas
    if columns == []:
        
        columnas_numericas = [
            col for col in df.columns
            if df[col].dtype in ["int64", "float64"] and col != target_col
        ]
    else:
        
        columnas_numericas = [
            col for col in columns
            if col in df.columns and df[col].dtype in ["int64", "float64"]
        ]

    # Si no hay columnas numéricas:
    if len(columnas_numericas) == 0:
        print("No hay columnas numéricas válidas para representar.")
        return []

    # 3. Usamos get_features_num_regression 
    # Creamos un DF reducido solo con target + columnas numéricas
    df_reducido = df[[target_col] + columnas_numericas]

    # Seleccionar variables relevantes
    seleccionadas = get_features_num_regression(
        df_reducido,
        target_col,
        umbral_corr,
        pvalue
    )

    # Si no hay variables seleccionadas:
    if seleccionadas is None:
        return None

    if len(seleccionadas) == 0:
        print("Ninguna variable numérica supera el umbral de correlación.")
        return []

    # 4. Realizamos los gráficos
       
    max_cols_plot = 5

    for i in range(0, len(seleccionadas), max_cols_plot - 1):       
        batch = seleccionadas[i:i + (max_cols_plot - 1)]

        cols_plot = [target_col] + batch

        # Creamos el pairplot
        sns.pairplot(df[cols_plot].dropna())
        plt.show()

  
    return seleccionadas


################ Siguiente función: get_features_cat_regression #######################


def get_features_cat_regression(df, target_col, pvalue=0.05):
    '''
    Descripción: esta función selecciona las columnas categóricas del dataframe cuya relación con la variable objetivo cumple ciertos
    criterios estadísticos y elige el tipo de test estadístico adecuado según el número de categorías.
    Argumentos:
    df (dataframe)
    target_col (columna): la columna objetivo, que debe ser numérica continua o discreta con alta cardinalidad.
    pvalue (float): entre 0 y 1.

    Retorna: una lista con los nombres de las columnas categóricas seleccionadas que cumplen los criterios de regresión.
    '''

    # primero comprueba si cada argumento que recibe es el indicado: 
    if not isinstance(df, (pd.DataFrame)):  # el tipo de objeto df debe ser un df; si no, printa el error
        print("Error: df debe ser un DataFrame de pandas")
        return None
    
    if not isinstance(target_col, str): # el tipo del objeto target_col debe ser str; si no, printa el error
        print("Error: target_col debe ser una cadena de texto")
        return None
    
    if target_col not in df.columns: # si la col target no está en las columnas del df, printa error
        print(f"Error: target_col '{target_col}' no existe en el dataframe")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]): # pd.api.types = función para comprobar el tipo de dato de una columna
        print("Error: target_col debe ser numérica para regresión") # si el tipo de dato de la col target_col no es de tipo numérico, printa error
        return None
    
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1): # si el tipo de objeto pvalue no es de tipo int/float o pvalue no está entre 0 y 1,
        print("Error: pvalue debe ser un número entre 0 y 1") # printa error
        return None
    
    # luego, vamos a buscar las variables categóricas:
    def get_cat_columns(df, target_col, max_unique_ratio=0.05):
        cat_cols = []
        n_rows = len(df)

        for col in df.columns:
            if col == target_col:
                continue

            nunique = df[col].nunique(dropna=True)
            dtype = df[col].dtype

            if (
                pd.api.types.is_object_dtype(dtype) or
                pd.api.types.is_string_dtype(dtype) or
                pd.api.types.is_categorical_dtype(dtype) or
                pd.api.types.is_bool_dtype(dtype)
            ):
                cat_cols.append(col)

            elif pd.api.types.is_numeric_dtype(dtype):
                if nunique / n_rows <= max_unique_ratio:
                    cat_cols.append(col)

        return cat_cols if cat_cols else None

    # Usamos la función para obtener las categóricas
    cat_cols = get_cat_columns(df, target_col)
    if cat_cols is None:
        print("No hay variables categóricas para analizar")
        return None
        
    # ahora vamos a iterar sobre cada columna categórica:
    cols_seleccionadas = [] # creamos nuestra lista vacía de las que nos valdrán

    for col in cat_cols:
        data = df[[col, target_col]].dropna()  # eliminamos filas con NaN
        grupos = [data[target_col][data[col] == cat] for cat in data[col].unique()]
        grupos = [g for g in grupos if len(g) > 1]  # eliminamos grupos con menos de 1 elemento
        n_cats = len(grupos)

        if n_cats < 2:
            # solo hay una categoría, no se puede testear
            continue

        try:
            if n_cats == 2:
                # t-test de Welch
                stat, p_val = stats.ttest_ind(grupos[0], grupos[1], equal_var=False)
            else:
                # ANOVA de un factor
                stat, p_val = stats.f_oneway(*grupos)
        except Exception as e:
            print(f"Advertencia: no se pudo evaluar la columna '{col}'. Motivo: {e}")
            continue

        # añadimos la columna a la lista SI pasa el p-value
        if p_val < pvalue:
            cols_seleccionadas.append(col)

    return cols_seleccionadas

############### Siguiente función: plot_features_cat_regression #######################

def plot_features_cat_regression(df, target_col="", pvalue=0.05, with_individual_plot=True):
    '''
    Descripción: esta función selecciona las columnas categóricas del dataframe cuya relación con la variable objetivo cumple
    ciertos criterios estadísticos, elige el tipo de test estadístico adecuado según el número de categorías, y genera histogramas
    del target para cada categoría de las columnas analizadas, marcando cuáles cumplen el criterio estadístico.

    Argumentos:
    df (dataframe)
    target_col (columna): la columna objetivo, que debe ser numérica continua o discreta con alta cardinalidad.
    pvalue (float): entre 0 y 1.
    with_individual_plot (bool): si True, genera histogramas del target por categoría.

    Retorna: una lista con los nombres de las columnas categóricas seleccionadas que cumplen los criterios de regresión.
    '''

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats

    # comprobaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: df debe ser un DataFrame de pandas")
        return None
    
    if not isinstance(target_col, str):
        print("Error: target_col debe ser una cadena de texto")
        return None
    
    if target_col not in df.columns:
        print(f"Error: target_col '{target_col}' no existe en el dataframe")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print("Error: target_col debe ser numérica para regresión")
        return None
    
    if not isinstance(pvalue, (int, float)) or not (0 < pvalue < 1):
        print("Error: pvalue debe ser un número entre 0 y 1")
        return None
    
    if not isinstance(with_individual_plot, bool):
        print("Error: with_individual_plot debe ser True o False")
        return None

    # función interna para detectar columnas categóricas
    def get_cat_columns(df, target_col, max_unique_ratio=0.05):
        cat_cols = []
        n_rows = len(df)

        for col in df.columns:
            if col == target_col:
                continue

            nunique = df[col].nunique(dropna=True)
            dtype = df[col].dtype

            # categóricas claras: object, string, categorical, bool
            if (
                pd.api.types.is_object_dtype(dtype) or
                pd.api.types.is_string_dtype(dtype) or
                pd.api.types.is_categorical_dtype(dtype) or
                pd.api.types.is_bool_dtype(dtype)
            ):
                cat_cols.append(col)

            # numéricas con baja cardinalidad
            elif pd.api.types.is_numeric_dtype(dtype):
                if nunique / n_rows <= max_unique_ratio:
                    cat_cols.append(col)

        return cat_cols if cat_cols else None

    # obtenemos las columnas categóricas
    cat_cols = get_cat_columns(df, target_col)
    if cat_cols is None:
        print("No hay variables categóricas para analizar")
        return None

    # lista para almacenar las columnas que pasan el test
    cols_seleccionadas = []

    # iteramos sobre cada columna categórica
    for col in cat_cols:
        data = df[[col, target_col]].dropna()  # eliminamos filas con NaN
        grupos = [data[target_col][data[col] == cat] for cat in data[col].unique()]
        grupos = [g for g in grupos if len(g) > 1]  # eliminamos grupos muy pequeños
        n_cats = len(grupos)

        if n_cats < 2:  # no se puede testear
            continue

        try:
            if n_cats == 2:
                # t-test de Welch
                stat, p_val = stats.ttest_ind(grupos[0], grupos[1], equal_var=False)
            else:
                # ANOVA de un factor
                stat, p_val = stats.f_oneway(*grupos)
        except Exception as e:
            print(f"Advertencia: no se pudo evaluar la columna '{col}'. Motivo: {e}")
            continue

        # añadimos a la lista si pasa el pvalue
        if p_val < pvalue:
            cols_seleccionadas.append(col)

# creamos los gráficos individuales con scatter y histograma del target
    if with_individual_plot and cols_seleccionadas:
        n = len(cols_seleccionadas)
        n_cols = min(3, n)  # máximo 3 columnas por fila
        n_rows = (n + n_cols - 1) // n_cols  # filas necesarias

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.array(axes).flatten()  # aplanamos en 1D para iterar

        for i, col in enumerate(cols_seleccionadas):
            data = df[[col, target_col]].dropna()
            nunique = data[col].nunique()

            if nunique <= 10:
                # pocas categorías → boxplot
                sns.boxplot(x=col, y=target_col, data=data, ax=axes[i], hue=None, palette="Set2", legend=False)
                axes[i].set_title(f"{col} vs {target_col}")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel(target_col)
            else:
                # muchas categorías → barplot de media de top N
                top_n = 10
                top_categories = data[col].value_counts().nlargest(top_n).index
                data_top = data[data[col].isin(top_categories)].copy()
                data_top[col] = data_top[col].astype(str)  # aseguramos tipo string
                mean_target = data_top.groupby(col)[target_col].mean().sort_values(ascending=False)
                sns.barplot(x=mean_target.index, y=mean_target.values, ax=axes[i], hue=None, palette="Set2", legend=False)
                axes[i].set_title(f"{col} (Top {top_n}) vs {target_col}")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel(f"Mean {target_col}")
                axes[i].tick_params(axis='x', rotation=45)

        # eliminamos ejes vacíos si los hay
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    if not cols_seleccionadas:
        print("Ninguna columna pasó el test de significancia")
        return []

    return cols_seleccionadas