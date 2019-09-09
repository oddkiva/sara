 #pragma once

 #include "OpenGLWindow.hpp"


class QOpenGLShaderProgram;

// class TriangleWindow : public OpenGLWindow
// {
// public:
//   TriangleWindow();
//
//   void initialize() override;
//   void render() override;
//
// private:
//   GLuint m_posAttr;
//   GLuint m_colAttr;
//   GLuint m_matrixUniform;
//
//   QOpenGLShaderProgram *m_program;
//   int m_frame;
// };
//! [1]

class TriangleWindow : public OpenGLWindow
{
public:
    TriangleWindow();

    void initialize() override;
    void render() override;

private:
    GLuint m_posAttr;
    GLuint m_colAttr;
    GLuint m_matrixUniform;

    QOpenGLShaderProgram *m_program;
    int m_frame;
};
//! [1]

