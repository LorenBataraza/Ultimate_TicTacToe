import glfw
from OpenGL.GL import *
import numpy as np

#from Ultimate_TicTacToe import * 

###############

# Inicializar GLFW
if not glfw.init():
    raise Exception("Fallo en la inicialización de GLFW")

# Crear una ventana de 800x600 píxeles con el título "Ventana OpenGL"
window = glfw.create_window(800, 600, "Super Tateti", None, None)
if not window:
    glfw.terminate()
    raise Exception("Fallo en la creación de la ventana")

# Hacer que el contexto de la ventana sea el actual
glfw.make_context_current(window)

# Defino color de fondo
glClearColor(0.4, 0.4, 0.4, 1.0)

# Definir los vértices de un triángulo (coordenadas x, y, z)
vertices = [-0.5, -0.5, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.0]
v = np.array(vertices, dtype=np.float32)

# Habilitar el uso de arrays de vértices
glEnableClientState(GL_VERTEX_ARRAY)
glVertexPointer(3, GL_FLOAT, 0, v)


def processInput(window):
    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
        glfw.set_window_should_close(window, True);


# Bucle principal: mientras la ventana no se cierre
while not glfw.window_should_close(window):
    # Procesar eventos (como cerrar la ventana)
    processInput(window)
    glfw.poll_events()

    # Limpiar el buffer de color
    glClear(GL_COLOR_BUFFER_BIT)
    
    # Dibujar el triángulo
    glDrawArrays(GL_TRIANGLES, 0, 3)
    
    # Intercambiar los buffers (mostrar lo renderizado)
    glfw.swap_buffers(window)

# Terminar GLFW y liberar recursos
glfw.terminate()