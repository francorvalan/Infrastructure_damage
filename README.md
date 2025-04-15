<h1 id="about"> :wrench: Aplicación para la segmentación y clasificación de daños en infraestructuras </h1>
<div align="center">
<!--     <a href="https://github.com/devicons/devicon">
        <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/streamlit/streamlit-original.svg" alt="Streamlit Logo" height="140" />
    </a> -->
    <h3 align="left">
        Transformar los sistemas de coordenas nunca habia sido tan sencillo
    </h3>
    <p align="center">
        <h2>
            <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/streamlit/streamlit-original.svg" 
                 alt="Demo2" 
                 height="60" /> <a href="https://j4tekdygvbbhuhdkulpkov.streamlit.app/" target="_blank">Demo</a>
        </h2>
    </p>

</div>
![SAM 2 architecture](Panel.png?raw=true)

La aplicacion permite clasificar e identificar daños en infraestructuras mediante el uso de [SAM2](https://github.com/facebookresearch/sam2).
Para ello es necesario:
        - 📷 Definir el directorio donde se encuentran las imagenes a procesar        
        - 🚩 Definir el tipo de infraestructura, tipo de daño y su grado
        - ⚙️ Elegir el trazado de la linea de la máscara 
        - 🤖 Indicar las directorios para el modelo y sus puntos de control

<h2 id="Run_locally">Correr localmente la aplicacion</h2>

1. Instalar las dependencias


   ```
   $ pip install -r requirements.txt
   ```
2. Ejecutar la aplicación

   ```
   $ streamlit run streamlit_app.py
   ```

<img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="80" /> <img src="https://cdn.jsdelivr.net/gh/devicons/devicon@latest/icons/streamlit/streamlit-original.svg" height="80" alt="streamlit logo"  style="margin-left: 12px;" />
