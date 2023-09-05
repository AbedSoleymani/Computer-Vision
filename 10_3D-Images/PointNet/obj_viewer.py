import pyglet
from pyglet.gl import *
import trimesh
import trimesh  # pip install trimesh

# Load the 3D OBJ file using trimesh
mesh = trimesh.load('./10_3D-Images/PointNet/stl_files/body.obj')

window = pyglet.window.Window(width=800, height=600, resizable=True)

@window.event
def on_draw():
    window.clear()
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glRotatef(45, 1, 1, 0)  # Rotate the mesh
    mesh.draw()

# Start the Pyglet event loop
pyglet.app.run()