#pragma once
#include <stdio.h>
#include <fstream>
#include "mpi_nnie.h"
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
#include "mpi_sys.h"
#endif
#include "hi_type.h"
#include "hi_nnie.h"
#include "hi_comm_svp.h"
#include "nnie_sys.h"
#include <opencv2/opencv.hpp>
/*16Byte align*/
#define SAMPLE_SVP_NNIE_ALIGN_16 16
#define SAMPLE_SVP_NNIE_ALIGN16(u32Num) ((u32Num + SAMPLE_SVP_NNIE_ALIGN_16-1) / SAMPLE_SVP_NNIE_ALIGN_16*SAMPLE_SVP_NNIE_ALIGN_16)
/*32Byte align*/
#define SAMPLE_SVP_NNIE_ALIGN_32 32
#define SAMPLE_SVP_NNIE_ALIGN32(u32Num) ((u32Num + SAMPLE_SVP_NNIE_ALIGN_32-1) / SAMPLE_SVP_NNIE_ALIGN_32*SAMPLE_SVP_NNIE_ALIGN_32)

#define SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(Type,Addr) (Type*)(HI_UL)(Addr)
#define SAMPLE_SVP_COORDI_NUM                     4        /*num of coordinates*/
#define SAMPLE_SVP_PROPOSAL_WIDTH                 6        /*the width of each proposal array*/
#define SAMPLE_SVP_QUANT_BASE                     4096     /*the basic quantity*/
#define SAMPLE_SVP_NNIE_EACH_SEG_STEP_ADDR_NUM    2

/*each seg input and output memory*/
typedef struct hiSAMPLE_SVP_NNIE_SEG_DATA_S
{
	SVP_SRC_BLOB_S astSrc[SVP_NNIE_MAX_INPUT_NUM];
	SVP_DST_BLOB_S astDst[SVP_NNIE_MAX_OUTPUT_NUM];
}SAMPLE_SVP_NNIE_SEG_DATA_S;
/*each seg input and output data memory size*/
typedef struct hiSAMPLE_SVP_NNIE_BLOB_SIZE_S
{
	HI_U32 au32SrcSize[SVP_NNIE_MAX_INPUT_NUM];
	HI_U32 au32DstSize[SVP_NNIE_MAX_OUTPUT_NUM];
}SAMPLE_SVP_NNIE_BLOB_SIZE_S;
/*NNIE input or output data index*/
typedef struct hiSAMPLE_SVP_NNIE_DATA_INDEX_S
{
	HI_U32 u32SegIdx;
	HI_U32 u32NodeIdx;
}SAMPLE_SVP_NNIE_DATA_INDEX_S;
/*this struct is used to indicate the input data from which seg's input or report node*/
typedef SAMPLE_SVP_NNIE_DATA_INDEX_S  SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S;
/*this struct is used to indicate which seg will be executed*/
typedef SAMPLE_SVP_NNIE_DATA_INDEX_S  SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S;

class NNIE_Net
{
public://fun
	NNIE_Net();
	~NNIE_Net();
	HI_S32 Read_weight_bias(HI_CHAR* buffer, HI_U64 size);

private:
	HI_S32 SAMPLE_COMM_SVP_NNIE_LoadModel(HI_CHAR * pszModelFile);

	HI_S32 SAMPLE_COMM_SVP_NNIE_LoadModel(HI_CHAR* buffer, HI_U64 size);
	HI_S32 SAMPLE_COMM_SVP_NNIE_ParamInit();
	HI_S32 SAMPLE_COMM_SVP_NNIE_ParamDeinit();
	//hardware
	HI_S32 SAMPLE_SVP_NNIE_ParamInit();
	//software
	//HI_S32 SAMPLE_SVP_NNIE_Rfcn_ParamInit(SAMPLE_SVP_NNIE_CFG_S* pstCfg, SAMPLE_SVP_NNIE_RFCN_SOFTWARE_PARAM_S* pstSoftWareParam);
	HI_S32 SAMPLE_SVP_NNIE_FillForwardInfo();
	HI_S32 SAMPLE_SVP_NNIE_GetTaskAndBlobBufSize(HI_U32*pu32TotalTaskBufSize, HI_U32*pu32TmpBufSize,
		SAMPLE_SVP_NNIE_BLOB_SIZE_S astBlobSize[], HI_U32*pu32TotalSize);
	HI_S32 SAMPLE_COMM_SVP_NNIE_UnloadModel();

protected:
	cv::Mat bias_matrix;
	cv::Mat weight_matrix;
	HI_S32 Read_weight_bias(HI_CHAR* matFile);
	cv::Mat Inner(HI_S32* indata, bool Relu);
	void Inner(HI_S32* indata, void* output, bool Relu);
	HI_S32 NNIE_NET_INIT(HI_CHAR * pszModelFile);
	HI_S32 NNIE_NET_INIT(HI_CHAR* buffer,HI_U64 size);

	HI_S32 SAMPLE_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx,
		SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S* pstProcSegIdx, HI_BOOL bInstant = HI_TRUE);
	HI_S32 SVP_FillSrcData_Mat(SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S* stInputDataIdx, const cv::Mat src);

	SVP_NNIE_MODEL_S    stModel;
	SVP_MEM_INFO_S      stModelBuf;//store Model file
	SVP_NNIE_ID_E aenNnieCoreId[SVP_NNIE_MAX_NET_SEG_NUM] = { SVP_NNIE_ID_0 };
	HI_U32 u32MaxInputNum=1;
	HI_U32 u32MaxRoiNum=0;//if not roi net ,it must to be 0
	HI_U64 au64StepVirAddr[SAMPLE_SVP_NNIE_EACH_SEG_STEP_ADDR_NUM*SVP_NNIE_MAX_NET_SEG_NUM];//virtual addr of LSTM's or RNN's step buffer

	HI_U32 Cal_stride(SVP_BLOB_TYPE_E type, HI_U32 width);

	//SVP_NNIE_MODEL_S*    pstModel;
	HI_U32 u32TmpBufSize;
	HI_U32 au32TaskBufSize[SVP_NNIE_MAX_NET_SEG_NUM];
	SVP_MEM_INFO_S      stTaskBuf;
	SVP_MEM_INFO_S      stTmpBuf;
	SVP_MEM_INFO_S      stStepBuf;//store Lstm step info
	SAMPLE_SVP_NNIE_SEG_DATA_S astSegData[SVP_NNIE_MAX_NET_SEG_NUM];//each seg's input and output blob
	SVP_NNIE_FORWARD_CTRL_S astForwardCtrl[SVP_NNIE_MAX_NET_SEG_NUM];
	SVP_NNIE_FORWARD_WITHBBOX_CTRL_S astForwardWithBboxCtrl[SVP_NNIE_MAX_NET_SEG_NUM];
};

