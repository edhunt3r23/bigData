import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
st.set_page_config(page_title="Análisis de Barritas de Proteína", layout="wide")

# CARGA Y LIMPIEZA DE DATOS
@st.cache_data
def cargar_datos():
    df = pd.read_csv("Barritas_Complete (1).csv", encoding="latin1")
    df.drop_duplicates(inplace=True)
    df.dropna(subset=["Price", "Region", "Product"], inplace=True)
    df.fillna("", inplace=True)
    df["Price"] = df["Price"].replace('[^\d.]', '', regex=True).astype(float)
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    return df

df = cargar_datos()
# Input decorativo de producto a analizar
st.title("Análisis de Productos")
st.markdown("Explora las tendencias del mercado, variables influyentes y predicciones de precios para barritas de proteína.")

producto_input = st.text_input("Ingresa el nombre del producto que te interesa:", "")
analizar = st.button("Analizar")

if analizar:
    if producto_input.strip() != "":
        st.success(f"Análisis iniciado")
        st.title("Dashboard de Barritas de Proteína")
        st.markdown("Explora las tendencias del mercado, variables influyentes y predicciones de precios para barritas de proteína.")

        # HIPÓTESIS 1: Precio promedio por tienda
        st.header("1. ¿Varía el precio promedio según la tienda?")
        st.markdown("Se analiza si algunas regiones venden más caro que otras.")
        avg_price = df.groupby("Region")["Price"].mean().sort_values()
        st.bar_chart(avg_price)

        # HIPÓTESIS 2: Correlación Precio - Cantidad
        st.header("2. ¿Existe una relación entre cantidad vendida y precio?")
        st.markdown("Se busca ver si hay relación entre lo que se vende y cuánto cuesta.")

        corr = df[["Price", "Quantity"]].corr().iloc[0, 1]
        st.write(f"**Correlación Precio - Cantidad:** {corr:.2f}")

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x="Quantity", y="Price", hue="Region", alpha=0.6)
        st.pyplot(fig)

        # HIPÓTESIS 3: ¿Qué tiendas venden más?
        st.header("3 ¿Qué tiendas (regiones) tienen más registros?")
        st.markdown("Se visualiza la cantidad de barritas registradas por región.")
        st.bar_chart(df["Region"].value_counts())

        # HIPÓTESIS 4: Modelo predictivo
        st.header("4 ¿Podemos predecir el precio usando más variables?")
        st.markdown("Incluimos también los **ingredientes** más comunes para mejorar el modelo.")

        # Preparamos ingredientes como variables
        if 'Ingredients' in df.columns:
            # Lista de los ingredientes más comunes
            ingredients_series = df['Ingredients'].str.lower().str.split(', ')
            all_ingredients = ingredients_series.explode()
            top_ingredients = all_ingredients.value_counts().head(20).index.tolist()

            for ing in top_ingredients:
                df[f"ing_{ing}"] = ingredients_series.apply(lambda x: int(ing in x if isinstance(x, list) else False))

        # Variables para el modelo
        cols_model = ["Price", "Quantity", "Region"] + [f"ing_{ing}" for ing in top_ingredients]
        df_model = df[cols_model].copy()

        # Dummies para Región
        df_model = pd.get_dummies(df_model, columns=["Region"], drop_first=True)
        df_model.dropna(subset=["Price", "Quantity"], inplace=True)

        X = df_model.drop("Price", axis=1)
        y = df_model["Price"]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # ---------- MODELO 1: Regresión Lineal ----------
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)

        r2_lr = r2_score(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))

        # ---------- MODELO 2: Random Forest ----------
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)

        r2_rf = r2_score(y_test, y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))

        # -------- Resultados ----------
        st.subheader(" Evaluación de modelos")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("###  Regresión Lineal")
            st.write(f"**R²:** {r2_lr:.4f}")
            st.write(f"**RMSE:** {rmse_lr:.2f}")
        with col2:
            st.markdown("###  Random Forest")
            st.write(f"**R²:** {r2_rf:.4f}")
            st.write(f"**RMSE:** {rmse_rf:.2f}")

        # Gráfico Real vs Predicho (Random Forest)
        st.subheader("Real vs Predicho (Random Forest)")

        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_rf, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel("Precio real")
        ax.set_ylabel("Precio predicho")
        st.pyplot(fig)

        # Importancia de características
        st.subheader("Importancia de variables (Random Forest)")
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(importances.head(15))


        # HIPÓTESIS 5: Variables que más influyen
        st.header("5 ¿Qué factores influyen más en el precio?")
        coefs = pd.Series(lr.coef_, index=X.columns).sort_values(ascending=False)
        st.bar_chart(coefs)

        # MATRIZ DE CORRELACIÓN
        st.header("Matriz de Correlación")
        st.markdown("Explora relaciones entre variables numéricas (precio, cantidad, ingredientes).")

        # Creamos una copia del DataFrame numérico con nombres abreviados
        num_df = df.select_dtypes(include=np.number).copy()
        abreviaciones = {col: col.replace("ing_", "")[:20] + "..." if len(col) > 20 else col.replace("ing_", "") for col in num_df.columns}
        num_df.rename(columns=abreviaciones, inplace=True)

        # Cálculo y visualización
        corr_matrix = num_df.corr()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5,
                    annot_kws={"fontsize": 8}, cbar_kws={'label': 'Correlación'}, ax=ax)

        ax.set_title("Matriz de Correlación", fontsize=14, pad=10)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        st.pyplot(fig)


        # Ingredientes más comunes
        if 'Ingredients' in df.columns:
            st.header("Ingredientes más comunes")
            st.markdown("Se extraen y cuentan los ingredientes más repetidos.")
            ingredients_series = df['Ingredients'].str.lower().str.split(', ')
            all_ingredients = ingredients_series.explode()
            top_ingredients = all_ingredients.value_counts().head(15)

            st.bar_chart(top_ingredients)

        #  Productos más populares por tienda
        st.header("Barritas más populares por tienda")
        st.markdown("Las barritas más frecuentes en cada tienda.")
        popular = df.groupby(["Region", "Product"]).size().reset_index(name="Count")
        top_by_region = popular.sort_values(["Region", "Count"], ascending=[True, False]).groupby("Region").head(1)
        st.dataframe(top_by_region)

        #  Clústeres de Barritas (Precio y Cantidad)
        st.header("Segmentación de barritas (Clustering)")
        st.markdown("Agrupamos productos similares en función de Precio y Cantidad.")

        cluster_df = df[["Price", "Quantity"]].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(cluster_df)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        cluster_df["Cluster"] = kmeans.fit_predict(X_scaled)

        fig, ax = plt.subplots()
        sns.scatterplot(data=cluster_df, x="Quantity", y="Price", hue="Cluster", palette="Set2")
        st.pyplot(fig)

        # Filtro interactivo y vista general
        st.header("Vista general de datos")
        st.markdown("Filtra los datos por tienda y rango de precios.")

        regions = st.multiselect("Selecciona regiones", options=df["Region"].unique(), default=df["Region"].unique())
        price_range = st.slider("Rango de precios", float(df["Price"].min()), float(df["Price"].max()), (float(df["Price"].min()), float(df["Price"].max())))

        df_filtered = df[(df["Region"].isin(regions)) & (df["Price"] >= price_range[0]) & (df["Price"] <= price_range[1])]
        st.dataframe(df_filtered)
