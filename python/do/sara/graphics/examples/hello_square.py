import sys

import numpy as np

import OpenGL.GL as gl

from PySide2.QtCore import QObject
from PySide2.QtGui import (QGuiApplication,
                           QOpenGLBuffer,
                           QOpenGLShader,
                           QOpenGLShaderProgram,
                           QOpenGLVertexArrayObject,
                           QOpenGLWindow,
                           QSurfaceFormat)


class SquareObject(QObject):

    def __init__(self, parent=None):
        super(SquareObject, self).__init__(parent=parent)
        self.initialize_shader_program()
        self.initialize_geometry()
        self.initialize_geometry_on_gpu()

    def initialize_shader_program(self):
        self.vertex_shader = """
            #version 330 core
              layout (location = 0) in vec3 in_coords;
              layout (location = 1) in vec3 in_color;

              out vec3 out_color;

              void main()
              {
                gl_Position = vec4(in_coords, 1.0);
                gl_PointSize = 200.0;
                out_color = in_color;
              }
        """

        self.fragment_shader = """
            #version 330 core
              in vec3 out_color;
              out vec4 frag_color;

              void main()
              {
                frag_color = vec4(out_color, 1.0);
              }
        """

        self.arg_pos = {
            'in_coords': 0,
            'in_color': 1,
            'out_color': 0
        }

        self.program = QOpenGLShaderProgram(parent=self.parent())
        self.program.addCacheableShaderFromSourceCode(QOpenGLShader.Vertex,
                                                      self.vertex_shader)
        self.program.addCacheableShaderFromSourceCode(QOpenGLShader.Fragment,
                                                      self.fragment_shader)
        self.program.link()

    def initialize_geometry(self):
        self.vertices = np.array([
            # coords           color
            [ 0.5,  0.5, 0.0,  1.0, 0.0, 0.0],  # top-right
            [ 0.5, -0.5, 0.0,  0.0, 1.0, 0.0],  # bottom-right
            [-0.5, -0.5, 0.0,  0.0, 0.0, 1.0],  # bottom-left
            [-0.5,  0.5, 0.0,  1.0, 1.0, 0.0]   # top-left
        ], dtype=np.float32)

        self.triangles = np.array([
            [0, 1, 2],
            [2, 3, 0]
        ], dtype=np.int32)

    def initialize_geometry_on_gpu(self):
        self.vao = QOpenGLVertexArrayObject(self.parent())
        if not self.vao.create():
            raise ValueError('Could not create VAO')
        self.vao.bind()

        self.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        if not self.vbo.create():
            raise ValueError('Could not create VBO')
        self.vbo.bind()
        self.vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        vertices_data = self.vertices.tostring()
        self.vbo.allocate(len(vertices_data))
        self.vbo.write(0, vertices_data, len(vertices_data))

        self.ebo = QOpenGLBuffer(QOpenGLBuffer.IndexBuffer)
        if not self.ebo.create():
            raise ValueError('Could not create EBO')
        self.ebo.bind()
        self.ebo.setUsagePattern(QOpenGLBuffer.StaticDraw)
        triangles_data = self.triangles.tostring()
        self.ebo.allocate(len(triangles_data))
        self.ebo.write(0, triangles_data, len(triangles_data))

        self.program.enableAttributeArray(self.arg_pos['in_coords'])
        self.program.setAttributeBuffer(self.arg_pos['in_coords'],
                                        gl.GL_FLOAT,
                                        0,
                                        3,
                                        self.vertices.shape[1] *
                                        self.vertices.dtype.itemsize)

        self.program.enableAttributeArray(self.arg_pos['in_color'])
        self.program.setAttributeBuffer(self.arg_pos['in_color'],
                                        gl.GL_FLOAT,
                                        3 * self.vertices.dtype.itemsize,
                                        3,
                                        self.vertices.shape[1] *
                                        self.vertices.dtype.itemsize)

        self.vao.release()
        self.vbo.release()
        self.ebo.release()

    def render(self):
        self.program.bind()
        self.vao.bind()
        gl.glDrawElements(gl.GL_TRIANGLES, self.triangles.size,
                          gl.GL_UNSIGNED_INT, None)
        self.program.release()


class Window(QOpenGLWindow):

    def initializeGL(self):
        super(Window, self).initializeGL()
        self.square = SquareObject(parent=self.context())

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_DEPTH_TEST)

    def paintGL(self):
        retinaScale = self.devicePixelRatio()
        gl.glViewport(0, 0, int(self.width() * retinaScale),
                      int(self.height() * retinaScale))

        gl.glClearColor(0.2, 0.3, 0.3, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.square.render()


if __name__ == '__main__':
    app = QGuiApplication(sys.argv)
    format = QSurfaceFormat()
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setVersion(3, 3)

    win = Window()
    win.setFormat(format)
    win.resize(800, 600)
    win.show()

    sys.exit(app.exec_())
