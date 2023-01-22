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

#include "NvEncoder/NvEncoderD3D12.h"

DXGI_FORMAT GetD3D12Format(NV_ENC_BUFFER_FORMAT eBufferFormat)
{
    switch (eBufferFormat)
    {
        case NV_ENC_BUFFER_FORMAT_ARGB:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        default:
            return DXGI_FORMAT_UNKNOWN;
    }
}

NvEncoderD3D12::NvEncoderD3D12(ID3D12Device* pD3D12Device, uint32_t nWidth, uint32_t nHeight, NV_ENC_BUFFER_FORMAT eBufferFormat, uint32_t nExtraOutputDelay) :
    NvEncoder(NV_ENC_DEVICE_TYPE_DIRECTX, pD3D12Device, nWidth, nHeight, eBufferFormat, nExtraOutputDelay, false, false, true)
{
    if (!pD3D12Device)
    {
        NVENC_THROW_ERROR("Bad D3D12device ptr", NV_ENC_ERR_INVALID_PTR);
        return;
    }

    if (GetPixelFormat() != NV_ENC_BUFFER_FORMAT_ARGB)
    {
        NVENC_THROW_ERROR("Unsupported Buffer format", NV_ENC_ERR_INVALID_PARAM);
    }

    if (!m_hEncoder)
    {
        NVENC_THROW_ERROR("Encoder Initialization failed", NV_ENC_ERR_INVALID_DEVICE);
    }

    m_pD3D12Device = pD3D12Device;
    m_nInputFenceVal = 0;
    m_nOutputFenceVal = 0;
}

NvEncoderD3D12::~NvEncoderD3D12()
{
    try
    {
        FlushEncoder();
        ReleaseOutputBuffers();
        ReleaseD3D12Resources();
    }
    catch (...)
    {

    }
}

void NvEncoderD3D12::AllocateInputBuffers(int32_t numInputBuffers)
{
    if (!IsHWEncoderInitialized())
    {
        NVENC_THROW_ERROR("Encoder intialization failed", NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
    }

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = GetMaxEncodeWidth();
    resourceDesc.Height = GetMaxEncodeHeight();
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;
    resourceDesc.Format = GetD3D12Format(GetPixelFormat());

    m_vInputBuffers.resize(numInputBuffers);

    for (int i = 0; i < numInputBuffers; i++)
    {
        if (m_pD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&m_vInputBuffers[i]))
                    != S_OK)
        {
            NVENC_THROW_ERROR("Failed to create ID3D12Resource", NV_ENC_ERR_OUT_OF_MEMORY);
        }
    }

    RegisterInputResources(GetMaxEncodeWidth(), GetMaxEncodeHeight(), GetPixelFormat());
    
    //Create NV_ENC_INPUT_RESOURCE_D3D12
    for (int i = 0; i < numInputBuffers; i++)
    {
        NV_ENC_INPUT_RESOURCE_D3D12* pInpRsrc = new NV_ENC_INPUT_RESOURCE_D3D12();
        memset(pInpRsrc, 0, sizeof(NV_ENC_INPUT_RESOURCE_D3D12));
        pInpRsrc->inputFencePoint.pFence = m_pInputFence.Get();

        m_vInputRsrc.push_back(pInpRsrc);
    }
}

void NvEncoderD3D12::ReleaseInputBuffers()
{
    if (!m_hEncoder)
    {
        return;
    }

    UnregisterInputResources();

    for (uint32_t i = 0; i < m_vInputRsrc.size(); ++i)
    {
         delete m_vInputRsrc[i];
    }
    m_vInputRsrc.clear();

    m_vInputFrames.clear();
}

void NvEncoderD3D12::RegisterInputResources(int width, int height, NV_ENC_BUFFER_FORMAT bufferFormat)
{
    for (uint32_t i = 0; i < m_vInputBuffers.size(); ++i)
    {
        NV_ENC_FENCE_POINT_D3D12 regRsrcInputFence;

        // Set input fence point
        memset(&regRsrcInputFence, 0, sizeof(NV_ENC_FENCE_POINT_D3D12));
        regRsrcInputFence.pFence = m_pInputFence.Get();
        regRsrcInputFence.waitValue = m_nInputFenceVal;
        regRsrcInputFence.bWait = true;

        InterlockedIncrement(&m_nInputFenceVal);

        regRsrcInputFence.signalValue = m_nInputFenceVal;
        regRsrcInputFence.bSignal = true;
        
        NV_ENC_REGISTERED_PTR registeredPtr = RegisterResource(m_vInputBuffers[i].Get(), NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, width, height, 0, bufferFormat, NV_ENC_INPUT_IMAGE,
            &regRsrcInputFence);

        NvEncInputFrame inputframe = {};
        ID3D12Resource *pTextureRsrc = m_vInputBuffers[i].Get();
        D3D12_RESOURCE_DESC desc = pTextureRsrc->GetDesc();
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT inputUploadFootprint[2];
        
        m_pD3D12Device->GetCopyableFootprints(&desc, 0, 1, 0, inputUploadFootprint, nullptr, nullptr, nullptr);

        inputframe.inputPtr = (void*)m_vInputBuffers[i].Get();
        inputframe.numChromaPlanes = NvEncoder::GetNumChromaPlanes(bufferFormat);
        inputframe.bufferFormat = bufferFormat;
        inputframe.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX;
        
        inputframe.pitch = inputUploadFootprint[0].Footprint.RowPitch;
        
        m_vRegisteredResources.push_back(registeredPtr);
        m_vInputFrames.push_back(inputframe);

        //CPU wait for register resource to finish
        CPUWaitForFencePoint((ID3D12Fence *)regRsrcInputFence.pFence, regRsrcInputFence.signalValue);
    }
}

void NvEncoderD3D12::AllocateOutputBuffers(uint32_t numOutputBuffers)
{
    HRESULT hr = S_OK;

    D3D12_RESOURCE_STATES  initialResourceState = D3D12_RESOURCE_STATE_COPY_DEST;

    D3D12_HEAP_PROPERTIES heapProps{};
    heapProps.Type = D3D12_HEAP_TYPE_READBACK;
    heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

    D3D12_RESOURCE_DESC resourceDesc{};
    resourceDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    resourceDesc.Alignment = 0;
    resourceDesc.Width = GetOutputBufferSize();
    resourceDesc.Height = 1;
    resourceDesc.DepthOrArraySize = 1;
    resourceDesc.MipLevels = 1;
    resourceDesc.Format = DXGI_FORMAT_UNKNOWN;
    resourceDesc.SampleDesc.Count = 1;
    resourceDesc.SampleDesc.Quality = 0;
    resourceDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    resourceDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

    m_vOutputBuffers.resize(numOutputBuffers);

    for (uint32_t i = 0; i < numOutputBuffers; i++)
    {
        if (m_pD3D12Device->CreateCommittedResource(&heapProps, D3D12_HEAP_FLAG_NONE, &resourceDesc, initialResourceState, nullptr, IID_PPV_ARGS(&m_vOutputBuffers[i])) != S_OK)
        {
            NVENC_THROW_ERROR("Failed to create output ID3D12Resource", NV_ENC_ERR_OUT_OF_MEMORY);
        }
    }
    
    RegisterOutputResources(GetOutputBufferSize());
    
    for (uint32_t i = 0; i < m_vOutputBuffers.size(); ++i)
    {
        NV_ENC_OUTPUT_RESOURCE_D3D12 *pOutRsrc = new NV_ENC_OUTPUT_RESOURCE_D3D12();
        memset(pOutRsrc, 0, sizeof(NV_ENC_OUTPUT_RESOURCE_D3D12));
        pOutRsrc->outputFencePoint.pFence = m_pOutputFence.Get();
        m_vOutputRsrc.push_back(pOutRsrc);
    }
}

void NvEncoderD3D12::ReleaseOutputBuffers()
{
    if (!m_hEncoder)
    {
        return;
    }

    UnregisterOutputResources();

    for (uint32_t i = 0; i < m_vOutputRsrc.size(); ++i)
    {
        delete m_vOutputRsrc[i];
    }
    m_vOutputRsrc.clear();
}

void NvEncoderD3D12::RegisterOutputResources(uint32_t bfrSize)
{
    for (uint32_t i = 0; i < m_vOutputBuffers.size(); ++i)
    {
        NV_ENC_REGISTERED_PTR registeredPtr = RegisterResource(m_vOutputBuffers[i].Get(), 
                                                      NV_ENC_INPUT_RESOURCE_TYPE_DIRECTX, bfrSize, 1, 0, 
                                                      NV_ENC_BUFFER_FORMAT_U8, NV_ENC_OUTPUT_BITSTREAM);
    
        m_vRegisteredResourcesOutputBuffer.push_back(registeredPtr);
    }
}

void NvEncoderD3D12::UnregisterOutputResources()
{
    for (uint32_t i = 0; i < m_vRegisteredResourcesOutputBuffer.size(); ++i)
    {
        if (m_vOutputRsrc[i])
        {
            m_nvenc.nvEncUnregisterResource(m_hEncoder, m_vRegisteredResourcesOutputBuffer[i]);
        }
    }
}

ID3D12Fence* NvEncoderD3D12::GetInpFence()
{
    return m_pInputFence.Get();
}

uint64_t* NvEncoderD3D12::GetInpFenceValPtr()
{
    return &m_nInputFenceVal;
}

void NvEncoderD3D12::CPUWaitForFencePoint(ID3D12Fence* pFence, uint64_t nFenceValue)
{
    if (pFence->GetCompletedValue() < nFenceValue)
    {
        if (pFence->SetEventOnCompletion(nFenceValue, m_event) != S_OK)
        {
            NVENC_THROW_ERROR("SetEventOnCompletion failed", NV_ENC_ERR_INVALID_PARAM);
        }
        WaitForSingleObject(m_event, INFINITE);
    }
}

uint32_t NvEncoderD3D12::GetInputSize()
{
    ID3D12Resource* pTexRsrc = (ID3D12Resource*)(m_vInputFrames[m_iToSend % m_nEncoderBuffer].inputPtr);
    D3D12_RESOURCE_DESC desc = pTexRsrc->GetDesc();
    uint64_t totalRequiredSizeInBytes = 0;
    
    m_pD3D12Device->GetCopyableFootprints(&desc, 0, 1, 0, nullptr, nullptr, nullptr, &totalRequiredSizeInBytes);

    return (uint32_t)totalRequiredSizeInBytes;
}

uint32_t NvEncoderD3D12::GetOutputBufferSize()
{
    uint32_t bufferSize = GetFrameSize() * 2;
    bufferSize = ALIGN_UP(bufferSize, 4);

    return bufferSize;
}
void NvEncoderD3D12::CreateEncoder(const NV_ENC_INITIALIZE_PARAMS* pEncoderParams)
{
    if (m_pD3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_pInputFence)) != S_OK)
    {
        NVENC_THROW_ERROR("Failed to create ID3D12Fence", NV_ENC_ERR_OUT_OF_MEMORY);
    }

    if (m_pD3D12Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_pOutputFence)) != S_OK)
    {    
        NVENC_THROW_ERROR("Failed to create ID3D12Fence", NV_ENC_ERR_OUT_OF_MEMORY);
    }

    m_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);

    NvEncoder::CreateEncoder(pEncoderParams);
    
    AllocateOutputBuffers(m_nEncoderBuffer);

    m_vMappedOutputBuffers.resize(m_nEncoderBuffer, nullptr);
}

void NvEncoderD3D12::ReleaseD3D12Resources()
{
    CloseHandle(m_event);
}

void NvEncoderD3D12::MapResources(uint32_t bfrIdx)
{
    NvEncoder::MapResources(bfrIdx);

    //map output surface
    NV_ENC_MAP_INPUT_RESOURCE mapInputResourceBitstreamBuffer = { NV_ENC_MAP_INPUT_RESOURCE_VER };
    mapInputResourceBitstreamBuffer.registeredResource = m_vRegisteredResourcesOutputBuffer[bfrIdx];
    NVENC_API_CALL(m_nvenc.nvEncMapInputResource(m_hEncoder, &mapInputResourceBitstreamBuffer));
    m_vMappedOutputBuffers[bfrIdx] = mapInputResourceBitstreamBuffer.mappedResource;
}

void NvEncoderD3D12::EncodeFrame(std::vector<std::vector<uint8_t>> &vPacket, NV_ENC_PIC_PARAMS *pPicParams)
{
    vPacket.clear();
    if (!IsHWEncoderInitialized())
    {
        NVENC_THROW_ERROR("Encoder device not found", NV_ENC_ERR_NO_ENCODE_DEVICE);
    }

    int bfrIdx = m_iToSend % m_nEncoderBuffer;

    MapResources(bfrIdx);
    
    InterlockedIncrement(&m_nOutputFenceVal);

    m_vOutputRsrc[bfrIdx]->pOutputBuffer = m_vMappedOutputBuffers[bfrIdx];
    m_vOutputRsrc[bfrIdx]->outputFencePoint.signalValue = m_nOutputFenceVal;
    m_vOutputRsrc[bfrIdx]->outputFencePoint.bSignal = true;
    
    m_vInputRsrc[bfrIdx]->pInputBuffer = m_vMappedInputBuffers[bfrIdx];
    m_vInputRsrc[bfrIdx]->inputFencePoint.waitValue = m_nInputFenceVal;
    m_vInputRsrc[bfrIdx]->inputFencePoint.bWait = true;
    
    NVENCSTATUS nvStatus = DoEncode(m_vInputRsrc[bfrIdx], m_vOutputRsrc[bfrIdx], pPicParams);

    if (nvStatus == NV_ENC_SUCCESS || nvStatus == NV_ENC_ERR_NEED_MORE_INPUT)
    {
        m_iToSend++;
        GetEncodedPacket(m_vOutputRsrc, vPacket, true);
    }
    else
    {
        NVENC_THROW_ERROR("nvEncEncodePicture API failed", nvStatus);
    }
}

void NvEncoderD3D12::EndEncode(std::vector<std::vector<uint8_t>> &vPacket)
{
    vPacket.clear();
    if (!IsHWEncoderInitialized())
    {
        NVENC_THROW_ERROR("Encoder device not initialized", NV_ENC_ERR_ENCODER_NOT_INITIALIZED);
    }

    SendEOS();

    GetEncodedPacket(m_vOutputRsrc, vPacket, false);
}

void NvEncoderD3D12::GetEncodedPacket(std::vector<NV_ENC_OUTPUT_RESOURCE_D3D12*>& vOutputBuffer, std::vector<std::vector<uint8_t>>& vPacket, bool bOutputDelay)
{
    unsigned int i = 0;
    int iEnd = bOutputDelay ? m_iToSend - m_nOutputDelay : m_iToSend;
    for (; m_iGot < iEnd; m_iGot++)
    {
        WaitForCompletionEvent(m_iGot % m_nEncoderBuffer);
        NV_ENC_LOCK_BITSTREAM lockBitstreamData = { NV_ENC_LOCK_BITSTREAM_VER };
        lockBitstreamData.outputBitstream = vOutputBuffer[m_iGot % m_nEncoderBuffer];
        lockBitstreamData.doNotWait = false;
        NVENC_API_CALL(m_nvenc.nvEncLockBitstream(m_hEncoder, &lockBitstreamData));

        uint8_t* pData = (uint8_t*)lockBitstreamData.bitstreamBufferPtr;
        if (vPacket.size() < i + 1)
        {
            vPacket.push_back(std::vector<uint8_t>());
        }
        vPacket[i].clear();
        if (m_initializeParams.encodeGUID == NV_ENC_CODEC_AV1_GUID)
        {
            if (m_bWriteIVFFileHeader)
            {
                m_IVFUtils.WriteFileHeader(vPacket[i], MAKE_FOURCC('A', 'V', '0', '1'), m_initializeParams.encodeWidth, m_initializeParams.encodeHeight, m_initializeParams.frameRateNum, m_initializeParams.frameRateDen, 0xFFFF);
                m_bWriteIVFFileHeader = false;
            }

            m_IVFUtils.WriteFrameHeader(vPacket[i], lockBitstreamData.bitstreamSizeInBytes, lockBitstreamData.outputTimeStamp);

        }
        vPacket[i].insert(vPacket[i].end(), &pData[0], &pData[lockBitstreamData.bitstreamSizeInBytes]);
        i++;

        NVENC_API_CALL(m_nvenc.nvEncUnlockBitstream(m_hEncoder, lockBitstreamData.outputBitstream));

        if (m_vMappedInputBuffers[m_iGot % m_nEncoderBuffer])
        {
            NVENC_API_CALL(m_nvenc.nvEncUnmapInputResource(m_hEncoder, m_vMappedInputBuffers[m_iGot % m_nEncoderBuffer]));
            m_vMappedInputBuffers[m_iGot % m_nEncoderBuffer] = nullptr;
        }

        if (m_vMappedOutputBuffers[m_iGot % m_nEncoderBuffer])
        {
            NVENC_API_CALL(m_nvenc.nvEncUnmapInputResource(m_hEncoder, m_vMappedOutputBuffers[m_iGot % m_nEncoderBuffer]));
            m_vMappedOutputBuffers[m_iGot % m_nEncoderBuffer] = nullptr;
        }
    }
}

void NvEncoderD3D12::FlushEncoder()
{
    if (!m_hEncoder)
    {
        return;
    }

    try
    {
        std::vector<std::vector<uint8_t>> pOutputBuffer;
        EndEncode(pOutputBuffer);
    }
    catch (...)
    {

    }
}

void NvEncoderD3D12::DestroyEncoder()
{
    if (!m_hEncoder)
    {
        return;
    }
    
    FlushEncoder();
    ReleaseOutputBuffers();

    NvEncoder::DestroyEncoder();
}
