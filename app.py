import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Título de la app
st.title('Análisis de precios de viviendas en Boston')

# Cargar el dataset
@st.cache_data
def load_data():
    return pd.read_csv('housing.csv')

data = load_data()

st.subheader('Vista previa de los datos')
st.dataframe(data.head())

# Visualización básica
st.subheader('Distribución de precios')
fig, ax = plt.subplots()
sns.histplot(data['MEDV'], kde=True, ax=ax)
st.pyplot(fig)

# Selección de variables
features = st.multiselect('Selecciona las variables predictoras:', options=list(data.columns.drop('MEDV')))

if features:
    X = data[features]
    y = data['MEDV']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    st.subheader('Resultados del modelo de regresión lineal')
    st.write(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    st.write(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
    st.line_chart(pd.DataFrame({'Real': y_test, 'Predicción': y_pred}).reset_index(drop=True))
else:
    st.info('Selecciona al menos una variable para entrenar el modelo.')
