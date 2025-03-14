import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

st.set_page_config(page_title='Eksplorasi Dataset Iris', layout='wide')
st.title('🌸 Aplikasi Eksplorasi Dataset Iris')
st.write('Menyajikan dataset Iris secara interaktif')

# Memuat dataset iris
st.header('📊 Dataset Iris')
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Sidebar
num_rows = st.sidebar.slider("Jumlah data yang ditampilkan", 5, len(df), 10)
st.dataframe(df.head(num_rows))

# Informasi Data
with st.expander('ℹ️ Informasi Dataset'):
    st.write("**Dimensi dataset:**", df.shape)
    st.write("**Deskripsi statistik:**")
    st.write(df.describe())
    st.write("**Missing values:**")
    st.write(df.isnull().sum())

# Visualisasi Data
st.header('📉 Visualisasi Data')
tab1, tab2 = st.tabs(['Line Plot Sepal Length', 'Line Plot Petal Length'])

# Tab 1 - Line Plot Sepal Length
with tab1:
    st.subheader("Line Plot - Sepal Length per Index")
    # Membuat grafik yg akan meng plot per indeks nya atau perdatanya
    # Sub plot menerima 2 variabel figure axe
    # Figure tempat menampilkan figure atau plotnya
    # Axes tempat mengedit figurenya
    fig, ax = plt.subplots(figsize=(10, 5))

    for species in df['target'].unique():
        subset = df[df['target'] == species]
        ax.plot(subset.index, subset['sepal length (cm)'], marker='o', label=iris.target_names[species])

    ax.set_xlabel("Index")
    ax.set_ylabel("Sepal Length (cm)")
    ax.set_title("Line Plot Sepal Length")
    ax.legend()
    st.pyplot(fig)

# Tab 2 - Line Plot Petal Length
with tab2:
    st.subheader("Line Plot - Petal Length per Index")
    # Menerima 2 variabel figure axe
    fig, ax = plt.subplots(figsize=(10, 5))

    for species in df['target'].unique():
        subset = df[df['target'] == species]
        ax.plot(subset.index, subset['petal length (cm)'], marker='o', label=iris.target_names[species])

    ax.set_xlabel("Index")
    ax.set_ylabel("Petal Length (cm)")
    ax.set_title("Line Plot Petal Length")
    ax.legend()
    st.pyplot(fig)
