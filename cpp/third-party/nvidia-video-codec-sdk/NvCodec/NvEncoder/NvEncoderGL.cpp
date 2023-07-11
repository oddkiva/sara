/*
 * This copyright notice applies to this header file only:
 *
 * Copyright (c) 2010-2023 NVIDIA Corporation
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the software, and to permit persons to whom the
 * software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "NvEncoder/NvEncoderGL.h"

NvEncoderGL::NvEncoderGL(uint32_t nWidth, uint32_t nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat,
    uint32_t nExtraOutputDelay, bool bMotionEstimationOnly) :
    NvEncoder(NV_ENC_DEVICE_TYPE_OPENGL, nullptr, nWidth, nHeight, eBufferFormat,
        nExtraOutputDelay, bMotionEstimationOnly)
{
    if (!m_hEncoder)
    {
        return;
    }
}

NvEncoderGL::~NvEncoderGL()
{
    ReleaseGLResources();
}

void NvEncoderGL::ReleaseInputBuffers()
{
    ReleaseGLResources();
}

void NvEncoderGL::AllocateInputBuffers(int32_t numInputBuffers)
{
    if (!IsHWEncoderInitialized())
    {
        NVENC_THROW_ERROR("Encoder device not initialized", NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
    }
    int numCount = m_bMotionEstimationOnly ? 2 : 1;

    for (int count = 0; count < numCount; count++)
    {
        std::vector<void*> inputFrames;
        for (int i = 0; i < numInputBuffers; i++)
        {
            NV_ENC_INPUT_RESOURCE_OPENGL_TEX *pResource = new NV_ENC_INPUT_RESOURCE_OPENGL_TEX;
            uint32_t tex;

            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_RECTANGLE, tex);

            uint32_t chromaHeight = GetNumChromaPlanes(GetPixelFormat()) * GetChromaHeight(GetPixelFormat(), GetMaxEncodeHeight());
            if (GetPixelFormat() == NV_ENC_BUFFER_FORMAT_YV12 || GetPixelFormat() == NV_ENC_BUFFER_FORMAT_IYUV)
                chromaHeight = GetChromaHeight(GetPixelFormat(), GetMaxEncodeHeight());

            glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_R8,
                GetWidthInBytes(GetPixelFormat(), GetMaxEncodeWidth()),
                GetMaxEncodeHeight() + chromaHeight,
                0, GL_RED, GL_UNSIGNED_BYTE, NULL);

            glBindTexture(GL_TEXTURE_RECTANGLE, 0);

            pResource->texture = tex;
            pResource->target = GL_TEXTURE_RECTANGLE;
            inputFrames.push_back(pResource);
        }
        RegisterInputResources(inputFrames, NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX,
            GetMaxEncodeWidth(),
            GetMaxEncodeHeight(),
            GetWidthInBytes(GetPixelFormat(), GetMaxEncodeWidth()),
            GetPixelFormat(), count == 1 ? true : false);
    }
}

void NvEncoderGL::ReleaseGLResources()
{
    if (!m_hEncoder)
    {
        return;
    }

    UnregisterInputResources();

    for (int i = 0; i < m_vInputFrames.size(); ++i)
    {
        if (m_vInputFrames[i].inputPtr)
        {
            NV_ENC_INPUT_RESOURCE_OPENGL_TEX *pResource = (NV_ENC_INPUT_RESOURCE_OPENGL_TEX *)m_vInputFrames[i].inputPtr;
            if (pResource)
            {
                glDeleteTextures(1, &pResource->texture);
                delete pResource;
            }
        }
    }
    m_vInputFrames.clear();

    for (int i = 0; i < m_vReferenceFrames.size(); ++i)
    {
        if (m_vReferenceFrames[i].inputPtr)
        {
            NV_ENC_INPUT_RESOURCE_OPENGL_TEX *pResource = (NV_ENC_INPUT_RESOURCE_OPENGL_TEX *)m_vReferenceFrames[i].inputPtr;
            if (pResource)
            {
                glDeleteTextures(1, &pResource->texture);
                delete pResource;
            }
        }
    }
    m_vReferenceFrames.clear();
}
