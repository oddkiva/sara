import sys

import numpy as np

import OpenGL.GL as gl

from PySide2 import QtGui


class TriangleWindow(QtGui.QOpenGLWindow):

    def __init__(self):
        super(TriangleWindow, self).__init__()

    def setup_shader_program(self):
        self._program = QtGui.QOpenGLShaderProgram(self.context())
        self._vertex_shader = """
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

        self._fragment_shader = """
        #version 330 core
  in vec3 out_color;
  out vec4 frag_color;

  void main()
  {
    vec2 circCoord = 2.0 * gl_PointCoord - 1.0;

    float dist = length(gl_PointCoord - vec2(0.5));

    if (dot(circCoord, circCoord) > 1.0)
        discard;
    float alpha = 1.0 - smoothstep(0.2, 0.5, dist);
    frag_color = vec4(out_color, alpha);
  }
        """

        self._arg_pos = {'in_coords': 0, 'in_color': 1, 'out_color': 0}

        self._program.addCacheableShaderFromSourceCode(
            QtGui.QOpenGLShader.Vertex, self._vertex_shader)
        self._program.addCacheableShaderFromSourceCode(
            QtGui.QOpenGLShader.Fragment, self._fragment_shader)

        self._vao = QtGui.QOpenGLVertexArrayObject(self.context())
        if not self._vao.create():
            raise QtCore.QException()

        self._vbo = QtGui.QOpenGLBuffer()
        if not self._vbo.create():
            raise QtCore.QException()

        self._data = np.array([
            # Coords             Color
            [-0.5, -0.5, 0.0,    1.0, 0.0, 0.0],  # left
            [ 0.5, -0.5, 0.0,    0.0, 1.0, 0.0],  # right
            [ 0.0,  0.5, 0.0,    0.0, 0.0, 1.0]   # top
        ], dtype=np.float32).tostring()

        self._program.bind()
        self._vao.bind()
        self._vbo.bind()
        self._vbo.setUsagePattern(QtGui.QOpenGLBuffer.StaticDraw)
        self._vbo.allocate(len(self._data))
        self._vbo.write(0, self._data, len(self._data))

        self._program.enableAttributeArray(self._arg_pos['in_coords'])
        self._program.setAttributeBuffer(self._arg_pos['in_coords'],
                                         gl.GL_FLOAT,
                                         0, 3, 6 * 4)

        self._program.enableAttributeArray(self._arg_pos['in_color'])
        self._program.setAttributeBuffer(self._arg_pos['in_color'],
                                         gl.GL_FLOAT,
                                         3 * 4, 3, 6 * 4)

    def initializeGL(self):
        super(TriangleWindow, self).initializeGL()
        self.setup_shader_program()
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

        self._program.bind()
        self._vao.bind()
        gl.glDrawArrays(gl.GL_POINTS, 0, 3)
        self._program.release()


if __name__ == '__main__':
    app = QtGui.QGuiApplication(sys.argv)

    format = QtGui.QSurfaceFormat()
    format.setProfile(QtGui.QSurfaceFormat.CoreProfile)
    format.setVersion(3, 3)

    win = TriangleWindow()
    win.setFormat(format)
    win.show()

    sys.exit(app.exec_())
