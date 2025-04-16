import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
# Deshabilitar el watcher problemático
st.runtime.installed_packages = []
import os
import numpy as np
from PIL import Image
import copy
import base64
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import asyncio
import sys
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec 
if sys.platform == "win32" and (3, 8, 0) <= sys.version_info < (3, 9, 0):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import torch
import urllib.request
import os


# Configuración inicial

OBJECTS = ['Daño', 'NA']
st.set_page_config(layout="wide")

# Clases y configuraciones predefinidas
TIPOS_DAÑO = ['Conexiones', 'Corrosión', 'Deformación','Grietas']
NIVELES_DAÑO = ['1', '2', '3', '4']
TIPOS_ESTRUCTURA = {
    'Cimientos y Pilotes': '4',
    'Columnas y Muros de carga': '4',
    'Vigas principales': '4',
    'Diagonales y Puntales': '3',
    'Losas': '3',
    'Dinteles': '3',
    'Columnas secundarias': '2',
    'Vigas secundarias': '2',
    'Soportes de escalerillas o ductos': '2',
    'Costaneras': '2',
    'Fundación': '4',
    'Revestimiento': '1',
    'Escaleras de acceso, pasarelas y barandas': '1',
    'Otros': '1'
}
FACTORES_CONTRIBUYENTE = ['Sin Factor','Altura', 'Sobrecarga']
GRUPOS=['Grupo_1','Grupo_2','Grupo_3','Grupo_4']

# Funciones
def show_mask(mask, ax, random_color=False, borders = True,label_col=None,alpha=0.1,fill=False,thickness=1,epsilon=1):
    if alpha==1:
        alpha=.99
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        
    if label_col is not None:
        color={
        '5':np.array([156, 0, 0 ,alpha]),
        '4':np.array([156, 0, 0 ,alpha]),
        '3': np.array([199, 76, 0 , alpha]),
        '2': np.array([222, 203, 0 , alpha]),
        '1': np.array([0, 148, 15 , alpha]),
        'NA': np.array([0, 255, 0 , 0])}[label_col]
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    

    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * (color/255).reshape(1, 1, -1)
    
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        
        contours = [cv2.approxPolyDP(contour, epsilon=epsilon, closed=True) for contour in contours]
        if label_col=='NA':
            mask_image = cv2.drawContours(mask_image, contours, -1, tuple(np.array([color[0]/255, color[1]/255, color[2]/255, 0])), thickness=thickness) 
        else:
            mask_image = cv2.drawContours(mask_image, contours, -1, tuple(np.array([color[0]/255, color[1]/255, color[2]/255, 1])), thickness=thickness) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=175):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()


def combinacion_de_mascaras(masks,Colors):
    ''' 
    Funcion para combinar mascaras de la mismas categoría
    '''
    masks = copy.deepcopy(masks)
    resultados_combinados=[]
    for color in np.unique(Colors):
        mascara_combinada=np.zeros_like(masks[0][0])
        # Índices de los grupos
        indices_col = [i for i, c in enumerate(Colors) if c == color]
        # Combinar las máscaras usando OR
        for idx in indices_col:
            mascara_combinada = np.logical_or(mascara_combinada, masks[0][idx])
        
        # Convertir a 0s y 1s y agregar a los resultados
        resultados_combinados.append(mascara_combinada.astype(int))
    
    # Reemplazar masks[0] con las máscaras combinadas
    masks[0] = resultados_combinados
    return(masks,np.unique(Colors))

def listar_imagenes_jpg(directorio):
    # Lista para almacenar las rutas de las imágenes .jpg
    imagenes_jpg = []

    # Recorrer el directorio y sus subdirectorios
    for raiz, directorios, archivos in os.walk(directorio):
        for archivo in archivos:
            # Verificar si el archivo tiene extensión .jpg (o .JPG)
            if archivo.lower().endswith('.png'):
                # Agregar la ruta completa del archivo a la lista
                imagenes_jpg.append(os.path.join(raiz, archivo))
    
    return imagenes_jpg

def combinacion_de_mascaras(masks, Colors):
    '''
    Función para combinar máscaras de la misma categoría.
    Si el color es "NA", se utiliza la máscara para cortar (cambiar 1 por 0) en el resto de las máscaras.
    '''
    # Crear una copia profunda de las máscaras para no modificar el original
    masks = copy.deepcopy(masks)
    
    # Lista para almacenar los resultados combinados
    resultados_combinados = []
    
    # Obtener la máscara correspondiente al color "NA"
    mascara_na = np.zeros_like(masks[0][0])  # Inicializar con ceros
    indices_na = [i for i, c in enumerate(Colors) if c == 'NA']
    for idx in indices_na:
        mascara_na = np.logical_or(mascara_na, masks[0][idx])
    
    # Agregar la máscara "NA" al final de los resultados
    resultados_combinados.append(mascara_na.astype(int))

    Colors_list=['NA']
    # Combinar máscaras para cada color único
    for color in np.unique(Colors):
        if color != "NA":
            # Inicializar una máscara vacía para este color
            mascara_combinada = np.zeros_like(masks[0][0])
            
            # Obtener los índices de las máscaras para este color
            indices_col = [i for i, c in enumerate(Colors) if c == color]
            
            # Combinar las máscaras usando OR
            for idx in indices_col:
                mascara_combinada = np.logical_or(mascara_combinada, masks[0][idx])
            
            # Aplicar la máscara "NA" para cortar (cambiar 1 por 0 donde la máscara "NA" es 1)
            mascara_combinada = np.where(mascara_na == 1, 0, mascara_combinada)
        
            # Convertir a 0s y 1s y agregar a los resultados
            resultados_combinados.append(mascara_combinada.astype(int))
            Colors_list.append(color)
            
    # Reemplazar masks[0] con las máscaras combinadas
    masks[0] = resultados_combinados
    
    # Devolver las máscaras combinadas y los colores únicos
    return masks, Colors_list

class Clase_Daño:
    # Constructor
    def __init__(self, tipo_daño, nivel_daño, tipo_estructura, nivel_estructura, factor_contribuyente):
        self.tipo_daño = tipo_daño
        self.nivel_daño = nivel_daño
        self.tipo_estructura = tipo_estructura
        self.nivel_estructura = nivel_estructura
        self.factor_contribuyente = factor_contribuyente
        
        # Calcular el nivel de compromiso
        self.nivel_compromiso = self.calcular_nivel_compromiso()
        
        # Determinar la clase de compromiso basada en el nivel de compromiso
        self.clase_compromiso = self.determinar_clase_compromiso()
        
        self.plan_de_accion = self.determinar_plan_de_accion()
    
    def calcular_nivel_compromiso(self):
        # Calcular el valor de compromiso
        compromiso = self.nivel_daño * self.nivel_estructura
        # Aplicar modificadores según el factor contribuyente
        if self.factor_contribuyente in ['Altura', 'Sobrecarga']:
            # Tabla de actualización para factores especiales
            actualizacion = {
                1: 2,
                2: 3,
                3: 4,
                4: 6,
                6: 8,
                8: 9,
                9: 12,
                12: 16,
                16: 17
            }
            compromiso = actualizacion.get(compromiso, compromiso)  # Devuelve el valor actualizado o el original si no está en la tabla
        
        # Determinar el nivel de compromiso
        if 1 <= compromiso <= 2:
            return 1
        elif 3 <= compromiso <= 5:
            return 2
        elif 6 <= compromiso <= 9:
            return 3
        elif 12 <= compromiso <= 16:
            return 4
        elif compromiso == 17:
            return 5
        else:
            return None  # En caso de que el valor no esté en los rangos definidos

    
    def determinar_plan_de_accion(self):
        planes_base = {
            1: 'Cada 12 meses',
            2: 'Cada 6 meses',
            3: 'Cada 3 meses',
            4: 'Menos de 1 mes',
            5: 'Menos de 1 mes'
        }
        
        planes_especiales = {
            'corrosión': {
                1: 'Cada 6 meses',
                2: 'Cada 3 meses',
                3: 'Cada 1 mes',
                4: 'Inmediato',
                5: 'Inmediato'
            },
        }
        
        tipo_daño = self.tipo_daño.lower()
        
        if tipo_daño in planes_especiales and self.nivel_compromiso in planes_especiales[tipo_daño]:
            return planes_especiales[tipo_daño][self.nivel_compromiso]
        
        return planes_base.get(self.nivel_compromiso, 'No definido') 

    def determinar_clase_compromiso(self):
        # Determinar la clase de compromiso basada en el nivel de compromiso
        if self.nivel_compromiso == 1:
            return 'Improbable'
        elif self.nivel_compromiso == 2:
            return 'Posible'
        elif self.nivel_compromiso == 3:
            return 'Probable'
        elif self.nivel_compromiso == 4:
            return 'Casi seguro'
        elif self.nivel_compromiso == 5:
            return 'Casi seguro'
        else:
            return 'Desconocido'  


def generar_figura_completa(image, best_masks2, Daño, colors2, Ancho_mascara):
    fig = plt.figure(figsize=(10, 8), dpi=100)
    gs_main = GridSpec(2, 1, height_ratios=[4, 1.5], hspace=0.05)
    
    # --- Parte superior: Imagen original con máscaras ---
    ax_main = fig.add_subplot(gs_main[0])
    ax_main.imshow(image)
    ax_main.axis('off')
    
    # Mostrar máscaras
    for i, mask in enumerate(best_masks2[0]):
        show_mask(
            mask, ax_main, 
            borders=True, label_col=colors2[i],
            alpha=0.01, fill=False,
            thickness=Ancho_mascara, epsilon=1
        )
    


    gs_leyenda = GridSpecFromSubplotSpec(
            1, 2, 
            subplot_spec=gs_main[1],  # Usar directamente la segunda fila
            width_ratios=[3, 7], 
            wspace=0.05
        )
    ax_leyenda_izq = fig.add_subplot(gs_leyenda[0])
    ax_leyenda_der = fig.add_subplot(gs_leyenda[1])
    
    ax_leyenda_izq.axis('off')
    ax_leyenda_der.axis('off')
    
    for ax in [ax_leyenda_izq, ax_leyenda_der]:
        ax.axis('off')

    matrix = [
        [4, 4, 8, 12, 16],
        [3, 3, 6, 9, 12],
        [2, 2, 4, 6, 8],
        [1, 1, 2, 3, 4],
        [np.nan, 1, 2, 3, 4]
    ]

    color_map = {
        1: '#119c41', 2: '#119c41', 3: '#f2f25a', 4: '#f2f25a',
        6: '#ffa629', 8: '#ffa629', 9: '#ffa629', 12: '#ff3e17',
        16: '#ff3e17', np.nan: 'white'
    }

    # tabla
    cell_colors = []
    for i, row in enumerate(matrix):
        new_row = []
        for j, val in enumerate(row):
            new_row.append('#d5e7f5' if j == 0 or i == 4 else color_map.get(val, 'white'))
        cell_colors.append(new_row)

    tabla = ax_leyenda_izq.table(
        cellText=[[str(x) if not np.isnan(x) else '' for x in row] for row in matrix],
        cellColours=cell_colors,
        cellLoc='center',
        loc='center',
        colWidths=[0.15]*5
    )

    for key, cell in tabla.get_celld().items():
        cell.set_height(0.18)
        cell.set_edgecolor('black')
        cell.set_linewidth(1.5)
        cell.set_text_props(fontsize=10)

    plt.draw() # Forzar renderizado antes de obtener coordenadas

    # Resaltar celda
    target_row = np.abs(Daño.nivel_daño - 4)
    target_col = Daño.nivel_estructura
    celda = tabla[target_row, target_col]
    x1, y1 = celda.get_xy()
    width, height = celda.get_width(), celda.get_height()
    center_x, center_y = x1 + width/2, y1 + height/2

    circulo = Circle(
        (center_x, center_y), 
        radius=width/2,
        color='black',
        fill=False,
        linewidth=3.5,
        zorder=3
    )
    ax_leyenda_izq.add_patch(circulo)
    circulo = Circle(
        (center_x, center_y),  
        radius=width/2,
        color='red',
        fill=False,
        linewidth=1.8,
        zorder=3
    )
    ax_leyenda_izq.add_patch(circulo)

    # Flechas y etiquetas
    for arrow_params, text_params in [
        ({'xy': (center_x, 0.05), 'xytext': (center_x, -0.10)}, 
         {'text': 'Función\nestructural', 'pos': (0.5, -0.2)}),
        ({'xy': (0.15, center_y), 'xytext': (0, center_y)}, 
         {'text': 'Daños', 'pos': (-0.05, 0.5), 'rotation': 90})
    ]:
        ax_leyenda_izq.annotate('', **arrow_params, 
            arrowprops=dict(arrowstyle='->', color='black', lw=1.3),
            annotation_clip=False)
        ax_leyenda_izq.text(*text_params['pos'], text_params['text'],
            ha='center', va='center', fontsize=9,
            rotation=text_params.get('rotation', 0),
            transform=ax_leyenda_izq.transAxes)

    # Texto descriptivo
    text_lines = [
        r'Tipo de infraestructura: $\mathbf{' + f'{Daño.tipo_estructura}'.replace(' ', r'\ ') + '}$',
    ]
    
    if Daño.factor_contribuyente in ['Altura', 'Sobrecarga']:
        text_lines.append(r'Factor contribuyente: $\mathbf{' + f'{Daño.factor_contribuyente}'.replace(' ', r'\ ') + '}$')
    
    text_lines.extend([
        r'Tipo de daño: $\mathbf{' + f'{Daño.tipo_daño}'.replace(' ', r'\ ') + '}$',
        r'Compromiso estructural: $\mathbf{' + f'{Daño.clase_compromiso}'.replace(' ', r'\ ') + '}$',
        r'Frecuencia: $\mathbf{' + f'{Daño.plan_de_accion}'.replace(' ', r'\ ') + '}$'
    ])

    ax_leyenda_der.text(
        0.0, 0.5, '\n'.join(text_lines),
        va='center', ha='left', fontsize=10, linespacing=1.0,
        bbox=dict(
            facecolor='#FEF9E7', edgecolor='#2E4053',
            linewidth=0.5, linestyle='-',
            boxstyle='round,pad=1.2,rounding_size=0.5', alpha=0.95
        )
    )
    
    return fig

#----------------------------------------------------------------------

def main():

    
    # Sidebar para controles
    with st.sidebar:
        st.header("Configuración de Procesamiento")
        
        # Selección de directorio
        input_dir = st.text_input("Directorio de imágenes:", value="C:/Users/francisco.corvalan/Codelco_ventana/02_Data/Imagenes/")
        
        # Cargar lista de imágenes
        if 'images' not in st.session_state or st.session_state.input_dir != input_dir:
            st.session_state.images = listar_imagenes_jpg(input_dir)
            st.session_state.input_dir = input_dir
            st.session_state.current_idx = 0
        
        # Navegación de imágenes
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Anterior") and st.session_state.current_idx > 0:
                st.session_state.current_idx -= 1
        with col2:
            if st.button("Siguiente") and st.session_state.current_idx < len(st.session_state.images)-1:
                st.session_state.current_idx += 1
                
        # Selección de atributos
        grupo_img = st.selectbox("Grupo", GRUPOS)
        tipo_daño = st.selectbox("Tipo de Daño", TIPOS_DAÑO)
        nivel_daño = st.selectbox("Nivel de daño:", NIVELES_DAÑO)
        tipo_estructura = st.selectbox("Tipo de estructura:", list(TIPOS_ESTRUCTURA.keys()))
        nivel_estructura = st.text_input("Nivel estructura:", value=TIPOS_ESTRUCTURA[tipo_estructura])
        factor_contribuyente = st.selectbox("Factor contribuyente:", FACTORES_CONTRIBUYENTE)
        Ancho_mascara = st.slider("Ancho máscara", 2, 6, 4)
        
        # Botón de procesamiento
        procesar = st.button("Procesar imagen actual")
        uploaded_model = st.file_uploader("Subir archivo de modelo (.yaml)", type=["yaml"])
        uploaded_checkpoint = st.file_uploader("Subir archivo de pesos (.pt)", type=["pt"])
        
        # 2. Guardar archivos temporalmente cuando se suban
        if uploaded_model and uploaded_checkpoint:
            # Crear carpeta temporal si no existe
            os.makedirs("temp_models", exist_ok=True)
            
            # Guardar archivos
            model_path = os.path.join("temp_models", "model.yaml")
            checkpoint_path = os.path.join("temp_models", "checkpoint.pt")
            
            with open(model_path, "wb") as f:
                f.write(uploaded_model.getbuffer())
            with open(checkpoint_path, "wb") as f:
                f.write(uploaded_checkpoint.getbuffer())
            
            st.success("¡Modelos cargados correctamente!")
            
            # 3. Inicializar el modelo solo si hay archivos nuevos
            if 'predictor' not in st.session_state:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                sam2_model = build_sam2(model_path, checkpoint_path, device=device)
                st.session_state.predictor = SAM2ImagePredictor(sam2_model)

    # Panel principal
    if st.session_state.images:
        current_image = st.session_state.images[st.session_state.current_idx]
        # Mostrar imagen original
        col1, col2 = st.columns(2)
        with col1:
            st.header("Seleccionar daños")
            img = Image.open(current_image)
            orig_width, orig_height = img.size
            display_width = 600
            scaling_factor = orig_width / display_width
            display_height = int(orig_height / scaling_factor)
            img_resized = img.resize((display_width, display_height))
            img_array_resized = np.array(img_resized.convert("RGB"))

            # Mostrar imagen redimensionada con scrollbars si es necesario
            click_data = streamlit_image_coordinates(img_array_resized, key="image", width=display_width)
            label = st.selectbox("Etiqueta", OBJECTS, key="current_label")

            if click_data is not None:
                if 'boxes' not in st.session_state:
                    st.session_state.boxes = []
                
                x_orig = int(click_data['x'] * scaling_factor)
                y_orig = int(click_data['y'] * scaling_factor)
                
                new_point = {
                    'x': x_orig,
                    'y': y_orig,
                    'width': 0,
                    'height': 0,
                    'label': st.session_state.current_label
                }
                st.session_state.boxes.append(new_point) 

            # Mostrar boxes en la imagen
            if 'boxes' in st.session_state and st.session_state.boxes:
                fig, ax = plt.subplots()
                ax.imshow(img_array_resized)  # Imagen redimensionada
                
                # Escalar coordenadas al tamaño de visualización
                scaling_factor = orig_width / display_width
                
                for box in st.session_state.boxes:
                    # Convertir coordenadas originales a escala de visualización
                    x_display = box['x'] / scaling_factor
                    y_display = box['y'] / scaling_factor
                    
                    if box['width'] == 0 and box['height'] == 0:
                        color = 'green' if box['label'] == 'Daño' else 'red'
                        ax.scatter(
                            x_display,  
                            y_display,
                            color=color, 
                            marker='o', 
                            s=50,
                            edgecolor='black'
                        )
                plt.axis('off')
                st.pyplot(fig)
                
            # limpiar selección
            if st.button("Limpiar selección"):
                if 'boxes' in st.session_state:
                    del st.session_state.boxes
                if 'pending_point' in st.session_state:
                    del st.session_state.pending_point
            
        # Procesar
        if procesar:
            with st.spinner("Procesando..."):
                # Validar que hay boxes seleccionados
                if 'boxes' not in st.session_state or len(st.session_state.boxes) == 0:
                    st.error("¡Debe seleccionar al menos un punto/área!")
                    return
                
                # Obtener los boxes de la sesión
                boxes = st.session_state.boxes
                Colors = [box['label'] for box in boxes]
                
                # Crear arrays para SAM
                points = np.array([[[box['x'], box['y']]] for box in boxes], dtype=np.float32)
                color_a_numero = {'Daño': 1, 'NA': 1}
                label_numeros = np.array([[color_a_numero[box['label']]] for box in boxes])
                
                # Cargar y procesar imagen
                image = np.array(img.convert("RGB"))
                st.session_state.predictor.set_image(image)
                
                # Crear instancia de daño
                daño = Clase_Daño(
                    tipo_daño=tipo_daño,
                    nivel_daño=int(nivel_daño),
                    tipo_estructura=tipo_estructura,
                    nivel_estructura=int(nivel_estructura) if nivel_estructura else 0,
                    factor_contribuyente=factor_contribuyente
                )
                
                # Generar máscaras con SAM2 usando los puntos seleccionados
                masks, scores, _ = st.session_state.predictor.predict(
                    point_coords=points,
                    point_labels=label_numeros,
                    multimask_output=True,
                )
                
                # Seleccionar mejores máscaras y combinar
                best_masks = []
                for mask, score in zip([masks], scores):
                    best_masks.append(mask[range(len(mask)), np.argmax(score, axis=-1)])
                
                best_masks2, colors2 = combinacion_de_mascaras(best_masks, Colors)
                colors2 =['NA',str(daño.nivel_compromiso)]
                
                # Generar y mostrar resultados
                with col2:
                    st.header("Imagen procesada")
                    fig = generar_figura_completa(image, best_masks2, daño, colors2, Ancho_mascara)
                    st.pyplot(fig)
                    # Guardar en disco (ajusta la ruta y nombre)

                    ruta_destino = f'C:/Users/francisco.corvalan/Codelco_ventana/03_Resultados/Imagenes_procesadas/{grupo_img}/{os.path.basename(current_image).replace('.','_processed.')}'
                    # Crear el directorio si no existe
                    directorio_destino = os.path.dirname(ruta_destino)
                    if not os.path.exists(directorio_destino):
                        os.makedirs(directorio_destino)

                    fig.savefig(
                        ruta_destino,
                        dpi=300,                
                        bbox_inches='tight',     
                        pad_inches=0.1           
                    )

                    # Cerrar la figura para liberar memoria
                    plt.close(fig)
                    st.success(f"Imagen guardada")
    else:
        st.warning("No se encontraron imágenes en el directorio especificado.")

if __name__ == "__main__":
    main()
