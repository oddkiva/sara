import sys
from os import path

import numpy as np

import OpenGL.GL as gl

from PySide2.QtCore import QObject, QElapsedTimer, QTimer
from PySide2.QtGui import (QGuiApplication,
                           QImage,
                           QMatrix4x4,
                           QOpenGLBuffer,
                           QOpenGLShader,
                           QOpenGLShaderProgram,
                           QOpenGLTexture,
                           QOpenGLVertexArrayObject,
                           QOpenGLWindow,
                           QSurfaceFormat,
                           QVector3D)


DATA_DIR = path.abspath('../../../../../data')


class CubeObject(QObject):

    def __init__(self, parent=None):
        super(CubeObject, self).__init__(parent=parent)
        self.initialize_shader_program()
        self.initialize_geometry()
        self.initialize_geometry_on_gpu()
        self.initialize_texture_on_gpu()

    def destroy(self):
        self.vao.release()
        self.vao.destroy()

        self.vbo.release()
        self.vbo.destroy()

        self.texture0.release()
        self.texture0.destroy()

        self.texture1.release()
        self.texture1.destroy()

    def initialize_shader_program(self):
        self.vertex_shader = """
            #version 330 core
              layout (location = 0) in vec3 in_coords;
              layout (location = 1) in vec2 in_tex_coords;

              uniform mat4 transform;
              uniform mat4 view;
              uniform mat4 projection;

              out vec3 out_color;
              out vec2 out_tex_coords;

              void main()
              {
                gl_Position = projection * view * transform * vec4(in_coords, 1.0);
                out_tex_coords = vec2(in_tex_coords.x, in_tex_coords.y);
              }
        """

        self.fragment_shader = """
            #version 330 core
              in vec3 out_color;
              in vec2 out_tex_coords;
              out vec4 frag_color;

              uniform sampler2D texture0;
              uniform sampler2D texture1;

              void main()
              {
                if (out_tex_coords.x > 0.5)
                  frag_color = texture(texture0, out_tex_coords);
                else
                  frag_color = texture(texture1, out_tex_coords);
                //frag_color = mix(texture(texture0, out_tex_coords),
                //                 texture(texture1, out_tex_coords), 0.5)
                //           * vec4(out_color, 1.0);
              }
        """

        self.arg_pos = {
            'in_coords': 0,
            'in_tex_coords': 1,
            'out_color': 0
        }

        self.program = QOpenGLShaderProgram(parent=self.parent())
        self.program.addCacheableShaderFromSourceCode(QOpenGLShader.Vertex,
                                                      self.vertex_shader)
        self.program.addCacheableShaderFromSourceCode(QOpenGLShader.Fragment,
                                                      self.fragment_shader)
        self.program.link()
        self.program.bind()

    def initialize_geometry(self):
        self.vertices = np.array([
            [-0.5, -0.5, -0.5,  0.0, 0.0],
            [ 0.5, -0.5, -0.5,  1.0, 0.0],
            [ 0.5,  0.5, -0.5,  1.0, 1.0],
            [ 0.5,  0.5, -0.5,  1.0, 1.0],
            [-0.5,  0.5, -0.5,  0.0, 1.0],
            [-0.5, -0.5, -0.5,  0.0, 0.0],

            [-0.5, -0.5,  0.5,  0.0, 0.0],
            [ 0.5, -0.5,  0.5,  1.0, 0.0],
            [ 0.5,  0.5,  0.5,  1.0, 1.0],
            [ 0.5,  0.5,  0.5,  1.0, 1.0],
            [-0.5,  0.5,  0.5,  0.0, 1.0],
            [-0.5, -0.5,  0.5,  0.0, 0.0],

            [-0.5,  0.5,  0.5,  1.0, 0.0],
            [-0.5,  0.5, -0.5,  1.0, 1.0],
            [-0.5, -0.5, -0.5,  0.0, 1.0],
            [-0.5, -0.5, -0.5,  0.0, 1.0],
            [-0.5, -0.5,  0.5,  0.0, 0.0],
            [-0.5,  0.5,  0.5,  1.0, 0.0],

            [ 0.5,  0.5,  0.5,  1.0, 0.0],
            [ 0.5,  0.5, -0.5,  1.0, 1.0],
            [ 0.5, -0.5, -0.5,  0.0, 1.0],
            [ 0.5, -0.5, -0.5,  0.0, 1.0],
            [ 0.5, -0.5,  0.5,  0.0, 0.0],
            [ 0.5,  0.5,  0.5,  1.0, 0.0],

            [-0.5, -0.5, -0.5,  0.0, 1.0],
            [ 0.5, -0.5, -0.5,  1.0, 1.0],
            [ 0.5, -0.5,  0.5,  1.0, 0.0],
            [ 0.5, -0.5,  0.5,  1.0, 0.0],
            [-0.5, -0.5,  0.5,  0.0, 0.0],
            [-0.5, -0.5, -0.5,  0.0, 1.0],

            [-0.5,  0.5, -0.5,  0.0, 1.0],
            [ 0.5,  0.5, -0.5,  1.0, 1.0],
            [ 0.5,  0.5,  0.5,  1.0, 0.0],
            [ 0.5,  0.5,  0.5,  1.0, 0.0],
            [-0.5,  0.5,  0.5,  0.0, 0.0],
            [-0.5,  0.5, -0.5,  0.0, 1.0]
        ], dtype=np.float32)

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

        self.program.enableAttributeArray(self.arg_pos['in_coords'])
        self.program.setAttributeBuffer(self.arg_pos['in_coords'],
                                        gl.GL_FLOAT,
                                        0,
                                        3,
                                        self.vertices.shape[1] *
                                        self.vertices.dtype.itemsize)

        self.program.enableAttributeArray(self.arg_pos['in_tex_coords'])
        self.program.setAttributeBuffer(self.arg_pos['in_tex_coords'],
                                        gl.GL_FLOAT,
                                        3 * self.vertices.dtype.itemsize,
                                        2,
                                        self.vertices.shape[1] *
                                        self.vertices.dtype.itemsize)

        self.vao.release()
        self.vbo.release()

    def initialize_texture_on_gpu(self):
        # Texture 0.
        image0 = QImage(path.join(DATA_DIR, 'ksmall.jpg')).mirrored()
        self.texture0 = QOpenGLTexture(image0)
        self.texture0.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
        self.texture0.setMagnificationFilter(QOpenGLTexture.Linear)
        self.texture0.setWrapMode(QOpenGLTexture.Repeat)
        self.texture0.bind(0)
        self.program.setUniformValue(self.program.uniformLocation('texture0'),
                                     0)

        # Texture 1.
        image1 = QImage(path.join(DATA_DIR, 'sunflowerField.jpg')).mirrored()
        self.texture1 = QOpenGLTexture(image1)
        self.texture1.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
        self.texture1.setMagnificationFilter(QOpenGLTexture.Linear)
        self.texture1.setWrapMode(QOpenGLTexture.Repeat)
        self.texture1.bind(1)
        self.program.setUniformValue(self.program.uniformLocation('texture1'),
                                     1)

    def render(self, transform):
        self.program.setUniformValue('transform', transform)
        self.vao.bind()
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.vertices.shape[0])


class Window(QOpenGLWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.timer = QElapsedTimer()
        self.timer.start()

        self.cubePositions = [
            QVector3D( 0.0,  0.0,   0.0),
            QVector3D( 2.0,  5.0, -15.0),
            QVector3D(-1.5, -2.2, - 2.5),
            QVector3D(-3.8, -2.0, -12.3),
            QVector3D( 2.4, -0.4, - 3.5),
            QVector3D(-1.7,  3.0, - 7.5),
            QVector3D( 1.3, -2.0, - 2.5),
            QVector3D( 1.5,  2.0, - 2.5),
            QVector3D( 1.5,  0.2, - 1.5),
            QVector3D(-1.3,  1.0, - 1.5)
        ]

    def initializeGL(self):
        super(Window, self).initializeGL()

        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_PROGRAM_POINT_SIZE)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.cube = CubeObject(parent=self.context())

    def paintGL(self):
        retinaScale = self.devicePixelRatio()
        gl.glViewport(0, 0, int(self.width() * retinaScale),
                      int(self.height() * retinaScale))

        gl.glClearColor(0.2, 0.3, 0.3, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        self.cube.program.bind()

        projection = QMatrix4x4()
        projection.setToIdentity()
        projection.perspective(45., float(self.width()) / self.height(), 0.1,
                               100.)
        self.cube.program.setUniformValue('projection', projection)

        # Initialize view matrix.
        view = QMatrix4x4()
        view.setToIdentity()
        view.translate(0., 0., -10.)
        self.cube.program.setUniformValue('view', view)

        for i, t in enumerate(self.cubePositions):
            transform = QMatrix4x4()
            transform.setToIdentity()
            transform.translate(t)
            transform.rotate(1.2 ** ((i + 1) * 5) * self.timer.elapsed() / 10,
                             QVector3D(0.5, 1, 0).normalized())

            self.cube.render(transform)

        self.cube.program.release()


if __name__ == '__main__':
    app = QGuiApplication(sys.argv)
    format = QSurfaceFormat()
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setVersion(3, 3)

    win = Window()
    win.setFormat(format)
    win.resize(800, 600)
    win.show()

    timer = QTimer()
    timer.start(20)
    timer.timeout.connect(win.update)

    def cleanup():
        global win
        win.makeCurrent()
        win.cube.destroy()
        win.doneCurrent()

    app.aboutToQuit.connect(cleanup)

    sys.exit(app.exec_())
