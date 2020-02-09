/*
* Copyright 2017-2018 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#pragma once

#include <iostream>
#include <algorithm>
#include <string.h>
#include <stdlib.h>
#include <mutex>
#include <thread>
#include <chrono>
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut_ext.h>
#include <cuda.h>
#include <cudaGL.h>
#include "../Utils/NvCodecUtils.h"
#include "../Utils/Logger.h"

class FramePresenterGL;
static FramePresenterGL *pInstance;

/**
* @brief OpenGL utility class to display decoded frames using OpenGL textures
*/
class FramePresenterGL
{
public:
    /**
    *   @brief  FramePresenterGL constructor function. This will launch a rendering thread which will be fed by decoded frames
    *   @param  cuContext - CUDA Context handle
    *   @param  nWidth - Width of OpenGL texture
    *   @param  nHeight - Height of OpenGL texture
    */
    FramePresenterGL(CUcontext cuContext, int nWidth, int nHeight) :
        cuContext(cuContext), nWidth(nWidth), nHeight(nHeight)
    {
        pthMessageLoop = new std::thread(ThreadProc, this);

        // This loop will ensure that OpenGL/glew/glut initialization is finished before we return from this constructor
        while (!pInstance) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    /**
    *   @brief  Destructor function. Also breaks glutMainLoopEvent
    */
    ~FramePresenterGL() {
        bStop = true;
        pthMessageLoop->join();
    }

    /**
    *   @brief  Writes CUdeviceptr handle which is later used for OpenGL rendering
    *   @param  pdpFrame - Pointer to frame data into which decoded frame is written
    *   @param  pnPitch - Pointer to pitch value. This function will write pitch value based on nWidth
    *   @return False if dpFrame is NULL or rendering loop is exited
    *   @return True when input params are successfully updated
    */
    bool GetDeviceFrameBuffer(uint8_t **pdpFrame, int *pnPitch) {
        if (bStop || !dpFrame) {
            return false;
        }
        
        *pdpFrame = (uint8_t *)dpFrame;
        *pnPitch = nWidth * 4;
        return true;
    }

    void SetText(std::string strText) {
        this->strText = strText;
    }

     /**
     *   @brief  Function to check whether underlying vendor is NVIDIA or not
     *   @return True if OpenGL vendor is NVIDIA
     *   @return False if OpenGL vendor is not NVIDIA
     */
     bool isVendorNvidia() {
        return this->vendorStatus;
    }

private:
     /**
     *   @brief  Thread launcher function. Launches the graphics initialization code
     *   @param  This - Pointer to FramePresenterGL object
     *   @return void
     */
     static void ThreadProc(FramePresenterGL *This) {
        This->Run();
    }

     /**
     *   @brief  Registered with glutDisplayFunc() as rendering function
     *   @return void
     */
     static void DisplayProc() {
        if (!pInstance) {
            return;
        }
        pInstance->Display();
    }

    static void CloseWindowProc() {
        if (!pInstance) {
            return;
        }
        // Enable the flag to break glutMainLoopEvent
        pInstance->bStop = true;
    }

    /**
    *   @brief  This function is responsible for OpenGL/glew/glut initialization and also for initiating display loop
    *   @return void
    */
    void Run() {
        int w = nWidth, h = nHeight;
        double r = (std::max)(nWidth / 1280.0, nHeight / 720.0);
        if (r > 1.0) {
            w = (int)(nWidth / r);
            h = (int)(nHeight / r);
        }

        int argc = 1;
        const char *argv[] = {"dummy"};
        glutInit(&argc, (char **)argv);
        glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
        glutInitWindowSize(w, h);
        glutCreateWindow("FramePresenterGL");
        glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

        char * vendor = (char*) glGetString(GL_VENDOR);

        if (!strcmp(vendor, "NVIDIA Corporation")) {
            this->vendorStatus = true;
        } else {
            // To prevent infinite loop in constructor, initialize pInstance
            pInstance = this;
            this->vendorStatus = false;
            return;
        }

        glViewport(0, 0, w, h);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);

        glewInit();
        // Prepare OpenGL buffer object for uploading texture data
        glGenBuffersARB(1, &pbo);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, nWidth * nHeight * 4, NULL, GL_STREAM_DRAW_ARB);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        // Create OpenGL texture and upload frame data
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex);
        glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGBA8, nWidth, nHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_RECTANGLE_ARB, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);

        // Create and initialize fragment program
        static const char *code =
            "!!ARBfp1.0\n"
            "TEX result.color, fragment.texcoord, texture[0], RECT; \n"
            "END";
        glGenProgramsARB(1, &shader);
        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
        glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB, (GLsizei)strlen(code), (GLubyte *)code);

        // Register display function
        glutDisplayFunc(DisplayProc);
        // Register window close event callback function
        glutCloseFunc(CloseWindowProc);
        
        ck(cuCtxSetCurrent(cuContext));
        ck(cuMemAlloc(&dpFrame, nWidth * nHeight * 4));
        ck(cuMemsetD8(dpFrame, 0, nWidth * nHeight * 4));

        pInstance = this;

        // Launch the rendering loop
        while (!bStop) {
            glutMainLoopEvent();
        }
        pInstance = NULL;

        ck(cuMemFree(dpFrame));

        glDeleteBuffersARB(1, &pbo);
        glDeleteTextures(1, &tex);
        glDeleteProgramsARB(1, &shader);
    }

    /**
    *   @brief  Rendering function called by glut
    *   @return void
    */
    void Display(void) {

        // Register the OpenGL buffer object with CUDA and map a CUdeviceptr onto it
        // The decoder code will receive this CUdeviceptr and copy the decoded frame into the associated device memory allocation
        CUgraphicsResource cuResource;
        ck(cuGraphicsGLRegisterBuffer(&cuResource, pbo, CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD));
        ck(cuGraphicsMapResources(1, &cuResource, 0));
        CUdeviceptr dpBackBuffer;
        size_t nSize = 0;
        ck(cuGraphicsResourceGetMappedPointer(&dpBackBuffer, &nSize, cuResource));

        CUDA_MEMCPY2D m = { 0 };
        m.srcMemoryType = CU_MEMORYTYPE_DEVICE;

        // Source is dpFrame into which Decode() function writes data of individual frame present in BGRA32 format
        // Destination is OpenGL buffer object mapped as a CUDA resource
        m.srcDevice = dpFrame;
        m.srcPitch = nWidth * 4;
        m.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        m.dstDevice = dpBackBuffer;
        m.dstPitch = nSize / nHeight;
        m.WidthInBytes = nWidth * 4;
        m.Height = nHeight;
        ck(cuMemcpy2DAsync(&m, 0));

        ck(cuGraphicsUnmapResources(1, &cuResource, 0));
        ck(cuGraphicsUnregisterResource(cuResource));

        // Bind OpenGL buffer object and upload the data
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, tex);
        glTexSubImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, 0, 0, nWidth, nHeight, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

        glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, shader);
        glEnable(GL_FRAGMENT_PROGRAM_ARB);
        glDisable(GL_DEPTH_TEST);

        // Specify vertex and texture co-ordinates
        glBegin(GL_QUADS);
        glTexCoord2f(0, (GLfloat)nHeight);
        glVertex2f(0, 0);
        glTexCoord2f((GLfloat)nWidth, (GLfloat)nHeight);
        glVertex2f(1, 0);
        glTexCoord2f((GLfloat)nWidth, 0);
        glVertex2f(1, 1);
        glTexCoord2f(0, 0);
        glVertex2f(0, 1);
        glEnd();
        glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
        glDisable(GL_FRAGMENT_PROGRAM_ARB);

        if (!strText.empty()) {
            PrintText(0, strText, 0, 0, true);
        }

        glutSwapBuffers();
        glutPostRedisplay();
    }

    static void PrintText(int iFont, std::string strText, int x, int y, bool bFillBackground) {
        struct {void *font; int d1; int d2;} fontData[] = {
            /*0*/ GLUT_BITMAP_9_BY_15,        13, 4,
            /*1*/ GLUT_BITMAP_8_BY_13,        11, 4,
            /*2*/ GLUT_BITMAP_TIMES_ROMAN_10, 9,  3,
            /*3*/ GLUT_BITMAP_TIMES_ROMAN_24, 20, 7,
            /*4*/ GLUT_BITMAP_HELVETICA_10,   10, 3,
            /*5*/ GLUT_BITMAP_HELVETICA_12,   11, 4,
            /*6*/ GLUT_BITMAP_HELVETICA_18,   16, 5,
        };
        const int nFont = sizeof(fontData) / sizeof(fontData[0]);

        if (iFont >= nFont) {
            iFont = 0;
        }
        void *font = fontData[iFont].font;
        int d1 = fontData[iFont].d1, d2 = fontData[iFont].d2, d = d1 + d2, 
            w = glutGet(GLUT_WINDOW_WIDTH), h = glutGet(GLUT_WINDOW_HEIGHT);

        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(0.0, w, 0.0, h, 0.0, 1.0);

        std::stringstream ss(strText);
        std::string str;
        int iLine = 0;
        while (std::getline(ss, str)) {
            glColor3f(1.0, 1.0, 1.0);
            if (bFillBackground) {
                glRasterPos2i(x, h - y - iLine * d - d1);
                for (char c : str) {
                    glutBitmapCharacter(font, c);
                }
                GLint pos[4];
                glGetIntegerv(GL_CURRENT_RASTER_POSITION, pos);
                glRecti(x, h - y - iLine * d, pos[0], h - y - (iLine + 1) * d);
                glColor3f(0.0, 0.0, 0.0);
            }
            glRasterPos2i(x, h - y - iLine * d - d1);
            for (char c : str) {
                glutBitmapCharacter(font, c);
            }
            iLine++;
        }

        glPopMatrix();
    }

private:
    bool bStop = false;         /*!< Variable to control rendering loop */
    int nWidth = 0, nHeight = 0;/*!< Dimensions of the frame being decoded */
    CUcontext cuContext = NULL;
    CUdeviceptr dpFrame = 0;    /*!< CUdeviceptr referenced by main thread to write decoded frames into */
    std::thread *pthMessageLoop = NULL;
    std::string strText;
    GLuint pbo;                 /*!< Buffer object to upload texture data */
    GLuint tex;                 /*!< OpenGL texture handle */
    GLuint shader;
    bool vendorStatus = false;  /*!< true if OpenGL vendor is NVIDIA, false otherwise */

};
