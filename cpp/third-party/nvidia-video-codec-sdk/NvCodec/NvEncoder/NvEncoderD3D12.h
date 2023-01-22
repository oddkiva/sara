/*
* Copyright 2022 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#pragma once

#include <vector>
#include <stdint.h>
#include <mutex>
#include <unordered_map>
#include <d3d12.h>
#include "NvEncoder.h"
#include <wrl.h>

#define ALIGN_UP(s,a) (((s) + (a) - 1) & ~((a) - 1))
using Microsoft::WRL::ComPtr;

class NvEncoderD3D12 : public NvEncoder
{
public:
    /**
    *  @brief  NvEncoderD3D12 class constructor.
    */
    NvEncoderD3D12(ID3D12Device* pD3D12Device, uint32_t nWidth, uint32_t nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat, 
        uint32_t nExtraOutputDelay = 3);

    /**
    *  @brief  NvEncoderD3D12 class virtual destructor.
    */
    virtual ~NvEncoderD3D12();

    /**
     *  @brief  This function is used to initialize the encoder session.
     *  Application must call this function to initialize the encoder, before
     *  starting to encode any frames.
     */
    void CreateEncoder(const NV_ENC_INITIALIZE_PARAMS* pEncoderParams) override;

    /**
     *  @brief  This function is used to get the input buffer size, needed 
     *  to allocate upload buffer.
     */
    uint32_t GetInputSize();

    /**
     *  @brief  This function is used to get the fence corresponding to 
     *  input buffers.
     */
    ID3D12Fence* GetInpFence();

    /**
     *  @brief  This function is used to get the pointer to fence value corresponding to
     *  input buffers.
     */
    uint64_t* GetInpFenceValPtr();

    /**
    *  @brief  This function is used to do CPU wait on fence and corresponding fence value.
    */
    void CPUWaitForFencePoint(ID3D12Fence* pFence, uint64_t nFenceValue);

    /**
     *  @brief  This function is used to get the total number of input buffers.
     */
    uint32_t GetNumBfrs() { return m_nEncoderBuffer;};

    /**
     *  @brief  This function is used to encode a frame.
     *  Applications must call EncodeFrame() function to encode the uncompressed
     *  data, which has been copied to an input buffer obtained from the
     *  GetNextInputFrame() function.
     */
    void EncodeFrame(std::vector<std::vector<uint8_t>>& vPacket, NV_ENC_PIC_PARAMS* pPicParams = nullptr) override;

    /**
    *  @brief  This function to flush the encoder queue.
    *  The encoder might be queuing frames for B picture encoding or lookahead;
    *  the application must call EndEncode() to get all the queued encoded frames
    *  from the encoder. The application must call this function before destroying
    *  an encoder session. Video memory buffer pointer containing compressed data
    *  is returned in pOutputBuffer.
    */
    void EndEncode(std::vector<std::vector<uint8_t>> &vPacket) override;

    /**
    *  @brief  This function is used to destroy the encoder session.
    *  Application must call this function to destroy the encoder session and
    *  clean up any allocated resources. The application must call EndEncode()
    *  function to get any queued encoded frames before calling DestroyEncoder().
    */
    void DestroyEncoder() override;

protected:
    /**
    *  @brief  This function is used to release the input buffers allocated for encoding.
    *  This function is an override of virtual function NvEncoder::ReleaseInputBuffers().
    */
    virtual void ReleaseInputBuffers() override;

private:
    /**
    *  @brief  This function is used to allocate input buffers for encoding.
    *  This function is an override of virtual function NvEncoder::AllocateInputBuffers().
    *  This function creates ID3D12Resource which is used to accept input data.
    *  To obtain handle to input buffers application must call NvEncoder::GetNextInputFrame()
    */
    virtual void AllocateInputBuffers(int32_t numInputBuffers) override;

    /**
     *  @brief  This function is used to allocate output buffers for storing encode output.
     */
    void AllocateOutputBuffers(uint32_t numOutputBuffers);

    /**
    *  @brief  This function is used to release output buffers.
    */
    void ReleaseOutputBuffers();

    /**
    *  @brief This function is used to map the input and output buffers to NvEncodeAPI.
    */
    void MapResources(uint32_t bfrIdx);

    /**
    *  @brief  This function is used to register output buffers with NvEncodeAPI.
    */
    void RegisterOutputResources(uint32_t bfrSize);

    /**
    *  @brief  This function is used to unregister output resources which had been previously registered for encoding
    *  using RegisterOutputResources() function.
    */
    void UnregisterOutputResources();

    /**
    *  @brief  This function is used to get the size of output buffer required to be
    *  allocated in order to store the output.
    */
    uint32_t GetOutputBufferSize();

    /**
    *  @brief  This function is used to register input buffers with NvEncodeAPI.
    */
    void RegisterInputResources(int width, int height, NV_ENC_BUFFER_FORMAT bufferFormat);

    /**
    *  @brief This is a private function which is used to get video memory buffer pointer containing compressed data
    *  from the encoder HW. This is called by EncodeFrame() function. If there is buffering enabled, this may return
    *  without any output data.
    */
    void GetEncodedPacket(std::vector<NV_ENC_OUTPUT_RESOURCE_D3D12*>& vOutputBuffer, std::vector<std::vector<uint8_t>>& vPacket, bool bOutputDelay);

    /**
    *  @brief  This is a private function to release ID3D12Texture2D textures used for encoding.
    */
    void ReleaseD3D12Resources();

    /**
    *  @brief  This function is used to flush the encoder queue.
    */
    void FlushEncoder();

protected:
    std::vector<NV_ENC_INPUT_RESOURCE_D3D12*> m_vInputRsrc;
    std::vector<NV_ENC_OUTPUT_RESOURCE_D3D12*> m_vOutputRsrc;

    ComPtr<ID3D12Fence> m_pInputFence;
    ComPtr<ID3D12Fence> m_pOutputFence;
    uint64_t m_nInputFenceVal;
    uint64_t m_nOutputFenceVal;
    HANDLE m_event;

    std::vector<ComPtr<ID3D12Resource>> m_vInputBuffers;
    std::vector<ComPtr<ID3D12Resource>> m_vOutputBuffers;

    std::vector<NV_ENC_REGISTERED_PTR> m_vRegisteredResourcesOutputBuffer;
    std::vector<NV_ENC_OUTPUT_PTR> m_vMappedOutputBuffers;

    ID3D12Device *m_pD3D12Device = nullptr;
};
