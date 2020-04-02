#include "NNIE_Net.h"
/*****************************************************************************
*   Prototype    : SAMPLE_SVP_NNIE_GetBlobMemSize
*   Description  : Get blob mem size
*   Input        : SVP_NNIE_NODE_S astNnieNode[]   NNIE Node
*                  HI_U32          u32NodeNum      Node num
*                  HI_U32          astBlob[]       blob struct
*                  HI_U32          u32Align        stride align type
*                  HI_U32          *pu32TotalSize  Total size
*                  HI_U32          au32BlobSize[]  blob size
*
*
*
*
*   Output       :
*   Return Value : VOID
*   Spec         :
*   Calls        :
*   Called By    :
*   History:
*
*       1.  Date         : 2017-11-20
*           Author       :
*           Modification : Create
*
*****************************************************************************/
static void SAMPLE_SVP_NNIE_GetBlobMemSize(SVP_NNIE_NODE_S astNnieNode[], HI_U32 u32NodeNum,
	HI_U32 u32TotalStep, SVP_BLOB_S astBlob[], HI_U32 u32Align, HI_U32* pu32TotalSize, HI_U32 au32BlobSize[])
{
	HI_U32 i = 0;
	HI_U32 u32Size = 0;
	HI_U32 u32Stride = 0;

	for (i = 0; i < u32NodeNum; i++)
	{
		if (SVP_BLOB_TYPE_S32 == astNnieNode[i].enType || SVP_BLOB_TYPE_VEC_S32 == astNnieNode[i].enType ||
			SVP_BLOB_TYPE_SEQ_S32 == astNnieNode[i].enType)
		{
			u32Size = sizeof(HI_U32);
		}
		else
		{
			u32Size = sizeof(HI_U8);
		}
		if (SVP_BLOB_TYPE_SEQ_S32 == astNnieNode[i].enType)
		{
			if (SAMPLE_SVP_NNIE_ALIGN_16 == u32Align)
			{
				u32Stride = SAMPLE_SVP_NNIE_ALIGN16(astNnieNode[i].unShape.u32Dim*u32Size);
			}
			else
			{
				u32Stride = SAMPLE_SVP_NNIE_ALIGN32(astNnieNode[i].unShape.u32Dim*u32Size);
			}
			au32BlobSize[i] = u32TotalStep*u32Stride;
		}
		else
		{
			if (SAMPLE_SVP_NNIE_ALIGN_16 == u32Align)
			{
				u32Stride = SAMPLE_SVP_NNIE_ALIGN16(astNnieNode[i].unShape.stWhc.u32Width*u32Size);
			}
			else
			{
				u32Stride = SAMPLE_SVP_NNIE_ALIGN32(astNnieNode[i].unShape.stWhc.u32Width*u32Size);
			}
			au32BlobSize[i] = astBlob[i].u32Num*u32Stride*astNnieNode[i].unShape.stWhc.u32Height*
				astNnieNode[i].unShape.stWhc.u32Chn;
		}
		*pu32TotalSize += au32BlobSize[i];
		astBlob[i].u32Stride = u32Stride;
	}
}
NNIE_Net::NNIE_Net()
{
}
NNIE_Net::~NNIE_Net()
{
	HI_S32 s32Ret;
	if (0 != stModelBuf.u64PhyAddr && 0 != stModelBuf.u64VirAddr) {
		SAMPLE_SVP_CHECK_EXPR_RET_VOID(stModel.u32NetSegNum > 8, SAMPLE_SVP_ERR_LEVEL_FATAL, "Error, u32NetSegNum >8 failed!\n");
		for (int i = 0; i < stModel.u32NetSegNum; i++)
		{
			s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&astForwardCtrl[i].stTskBuf);
			SAMPLE_SVP_CHECK_EXPR_RET_VOID(0 != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");
			if (stModel.astSeg[i].enNetType == SVP_NNIE_NET_TYPE_RECURRENT)
				if (0 != stStepBuf.u64PhyAddr && 0 != stStepBuf.u64VirAddr)
				{
					SvpSampleFree(stStepBuf.u64PhyAddr, (HI_VOID *)stStepBuf.u64VirAddr);
					stStepBuf.u64PhyAddr = 0;
					stStepBuf.u64VirAddr = 0;
				}
		}
		s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit();
		SAMPLE_SVP_CHECK_EXPR_RET_VOID(0 != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
		//********************* release model
		s32Ret = SAMPLE_COMM_SVP_NNIE_UnloadModel();
		SAMPLE_SVP_CHECK_EXPR_RET_VOID(0 != s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, SAMPLE_COMM_SVP_NNIE_UnloadModel failed!\n");
	}
	if (!weight_matrix.empty())
		weight_matrix.release();
	if (!bias_matrix.empty())
		bias_matrix.release();
}

HI_S32 NNIE_Net::Read_weight_bias(HI_CHAR * buffer, HI_U64 size)
{
	if (this->stModel.u32NetSegNum <= 1)
		return HI_FAILURE;
	int* buffer4 = (int*)buffer;
	int w = buffer4[0];
	int h = buffer4[1];
	if (size != (w*h + h * 1 + 2) * sizeof(float))
		return HI_FAILURE;
	this->bias_matrix = cv::Mat(h, 1, CV_32FC1, (float*)buffer4 + 2);
	this->weight_matrix = cv::Mat(h, w, CV_32FC1, (float*)buffer4 + 2 + h);
	return HI_SUCCESS;
}

HI_S32 NNIE_Net::Read_weight_bias(HI_CHAR * paramfile)
{
	if (this->stModel.u32NetSegNum <= 1)
		return HI_FAILURE;
	std::ifstream param(paramfile, std::ios::binary);
	int w, h;
	param.read((char *)&w, sizeof(int));
	param.read((char *)&h, sizeof(int));
	float *bias_v = new float[h];
	float *weight_v = new float[w*h];
	param.read((char *)bias_v, sizeof(float)*h);
	param.read((char *)weight_v, sizeof(float)*h*w);
	this->bias_matrix = cv::Mat(h, 1, CV_32FC1, bias_v);
	this->weight_matrix = cv::Mat(h, w, CV_32FC1, weight_v);
	param.close();
	return HI_SUCCESS;
}

cv::Mat NNIE_Net::Inner(HI_S32 * indata, bool Relu)
{
	cv::Mat seg0out, absf, seg1_in, w_1_m;
	seg0out = cv::Mat(this->weight_matrix.cols, 1, CV_32SC1, indata);
	cv::Mat seg0out_f(seg0out.size(), CV_32FC1);
	seg0out.convertTo(seg0out_f, CV_32FC1, 1.0 / 4096, 0);
	w_1_m = this->weight_matrix*seg0out_f + this->bias_matrix;
	if (!Relu) {
		w_1_m.convertTo(seg1_in, CV_32SC1, 4096.0, 0);
	}
	else {
		absf = abs(w_1_m);
		w_1_m += absf;
		w_1_m.convertTo(seg1_in, CV_32SC1, 2048.0, 0);
	}
	return seg1_in;
}

void NNIE_Net::Inner(HI_S32* indata,void* output, bool Relu)
{
cv::Mat seg1_in(bias_matrix.size(),CV_32SC1,output);
	cv::Mat seg0out, absf, w_1_m;
	seg0out = cv::Mat(this->weight_matrix.cols, 1, CV_32SC1, indata);
	cv::Mat seg0out_f(seg0out.size(), CV_32FC1);
	seg0out.convertTo(seg0out_f, CV_32FC1, 1.0 / 4096, 0);
	w_1_m = this->weight_matrix*seg0out_f + this->bias_matrix;
	if (!Relu) {
		w_1_m.convertTo(seg1_in, CV_32SC1, 4096.0, 0);
	}
	else {
		absf = abs(w_1_m);
		w_1_m += absf;
		w_1_m.convertTo(seg1_in, CV_32SC1, 2048.0, 0);
	}
	//return seg1_in;
}


HI_S32 NNIE_Net::NNIE_NET_INIT(HI_CHAR * pszModelFile)
{
	HI_S32 s32Ret;
	s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pszModelFile);

	SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");
	s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit();
	//printf("return = %d\n", s32Ret);
	SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n");
	return HI_SUCCESS;
}

HI_S32 NNIE_Net::NNIE_NET_INIT(HI_CHAR * buffer, HI_U64 size)
{
	HI_S32 s32Ret;
	s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(buffer,size);
	SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");
	s32Ret = SAMPLE_COMM_SVP_NNIE_ParamInit();
	SAMPLE_SVP_CHECK_EXPR_RET(0 != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, SAMPLE_COMM_SVP_NNIE_ParamInit failed!\n");
	return HI_SUCCESS;
}
HI_S32 NNIE_Net::SAMPLE_COMM_SVP_NNIE_LoadModel(HI_CHAR * buffer, HI_U64 size)
{
	HI_S32 s32Ret = 0;
	HI_U64 u64PhyAddr = 0;
	HI_U8 *pu8VirAddr = NULL;
	HI_S32 slFileSize = 0;
	/*malloc model file mem*/
	s32Ret = SAMPLE_COMM_SVP_MallocMem((HI_CHAR*)"SAMPLE_NNIE_MODEL", NULL, (HI_U64*)&u64PhyAddr, (void**)&pu8VirAddr, size);
	SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error(%#x),Malloc memory failed!\n", s32Ret);

	stModelBuf.u32Size = (HI_U32)size;
	stModelBuf.u64PhyAddr = u64PhyAddr;
	stModelBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;
	memcpy(pu8VirAddr, buffer, size);
	/*load model*/
	s32Ret = HI_MPI_SVP_NNIE_LoadModel(&stModelBuf, &stModel);
	SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,HI_MPI_SVP_NNIE_LoadModel failed!\n");
	//printf("buffer LOAD SUCESS\n");
	return s32Ret;
FAIL_1:
	printf("LOAD FAILED\n");
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	SAMPLE_SVP_MMZ_FREE(stModelBuf.u64PhyAddr, stModelBuf.u64VirAddr);
	stModelBuf.u32Size = 0;
#else
	SvpSampleMemFree(&stModelBuf);
#endif

FAIL_0:
	return HI_FAILURE;
}
HI_S32 NNIE_Net::SAMPLE_COMM_SVP_NNIE_LoadModel(HI_CHAR * pszModelFile)
{
	HI_S32 s32Ret = 0;
	HI_U64 u64PhyAddr = 0;
	HI_U8 *pu8VirAddr = NULL;
	HI_S32 slFileSize = 0;
	/*Get model file size*/
	FILE *fp = fopen(pszModelFile, "rb");
	SAMPLE_SVP_CHECK_EXPR_RET(NULL == fp, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, open model file failed!\n");
	s32Ret = fseek(fp, 0L, SEEK_END);
	SAMPLE_SVP_CHECK_EXPR_GOTO(-1 == s32Ret, FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, fseek failed!\n");
	slFileSize = ftell(fp);
	SAMPLE_SVP_CHECK_EXPR_GOTO(slFileSize <= 0, FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, ftell failed!\n");
	s32Ret = fseek(fp, 0L, SEEK_SET);
	SAMPLE_SVP_CHECK_EXPR_GOTO(-1 == s32Ret, FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, fseek failed!\n");

	/*malloc model file mem*/
	s32Ret = SAMPLE_COMM_SVP_MallocMem((HI_CHAR*)"SAMPLE_NNIE_MODEL", NULL, (HI_U64*)&u64PhyAddr, (void**)&pu8VirAddr, slFileSize);
	SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, FAIL_0, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error(%#x),Malloc memory failed!\n", s32Ret);

	stModelBuf.u32Size = (HI_U32)slFileSize;
	stModelBuf.u64PhyAddr = u64PhyAddr;
	stModelBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;

	s32Ret = fread(pu8VirAddr, slFileSize, 1, fp);
	SAMPLE_SVP_CHECK_EXPR_GOTO(1 != s32Ret, FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,read model file failed!\n");

	/*load model*/
	s32Ret = HI_MPI_SVP_NNIE_LoadModel(&stModelBuf, &stModel);
	SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,HI_MPI_SVP_NNIE_LoadModel failed!\n");

	fclose(fp);
	//printf("LOAD SUCESS\n");
	return s32Ret;
FAIL_1:
	printf("LOAD FAILED\n");
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	SAMPLE_SVP_MMZ_FREE(stModelBuf.u64PhyAddr, stModelBuf.u64VirAddr);
	stModelBuf.u32Size = 0;
#else
	SvpSampleMemFree(&stModelBuf);
#endif

FAIL_0:
	if (NULL != fp)
	{
		fclose(fp);
	}
	return HI_FAILURE;
}
HI_S32 NNIE_Net::SAMPLE_COMM_SVP_NNIE_UnloadModel()
{
	HI_S32 u32Ret;
	u32Ret = HI_MPI_SVP_NNIE_UnloadModel(&stModel);
	SAMPLE_SVP_CHECK_EXPR_RET(0 != u32Ret, u32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error, UnloadModel failed!\n");
	if (0 != stModelBuf.u64PhyAddr && 0 != stModelBuf.u64VirAddr)
	{
		SvpSampleFree(stModelBuf.u64PhyAddr, (HI_VOID*)stModelBuf.u64VirAddr);
		stModelBuf.u64PhyAddr = 0;
		stModelBuf.u64VirAddr = 0;
	}
	return HI_SUCCESS;
}

HI_S32 NNIE_Net::SAMPLE_COMM_SVP_NNIE_ParamInit()
{
	HI_S32 s32Ret = HI_SUCCESS;
	/*NNIE parameter initialization */
	s32Ret = SAMPLE_SVP_NNIE_ParamInit();
	SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, FAIL, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error, SAMPLE_SVP_NNIE_ParamInit failed!\n");

	return s32Ret;
FAIL:
	s32Ret = SAMPLE_COMM_SVP_NNIE_ParamDeinit();
	SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error, SAMPLE_COMM_SVP_NNIE_ParamDeinit failed!\n");
	return HI_FAILURE;
}

HI_S32 NNIE_Net::SAMPLE_COMM_SVP_NNIE_ParamDeinit()
{
	//SAMPLE_SVP_CHECK_EXPR_RET(NULL == pstNnieParam, HI_INVALID_VALUE, SAMPLE_SVP_ERR_LEVEL_ERROR,
	//	"Error, pstNnieParam can't be NULL!\n");

	if (0 != stTaskBuf.u64PhyAddr && 0 != stTaskBuf.u64VirAddr)
	{
		SvpSampleFree(stTaskBuf.u64PhyAddr, (HI_VOID *)stTaskBuf.u64VirAddr);
		stTaskBuf.u64PhyAddr = 0;
		stTaskBuf.u64VirAddr = 0;
	}

	return HI_SUCCESS;
}

HI_S32 NNIE_Net::SAMPLE_SVP_NNIE_ParamInit()
{
	HI_U32 i = 0, j = 0;
	HI_U32 u32TotalSize = 0;
	HI_U32 u32TotalTaskBufSize = 0;
	HI_U32 u32TmpBufSize = 0;
	HI_S32 s32Ret = HI_SUCCESS;
	HI_U32 u32Offset = 0;
	HI_U64 u64PhyAddr = 0;
	HI_U8 *pu8VirAddr = NULL;
	SAMPLE_SVP_NNIE_BLOB_SIZE_S astBlobSize[SVP_NNIE_MAX_NET_SEG_NUM] = { 0 };

	/*fill forward info*/
	s32Ret = SAMPLE_SVP_NNIE_FillForwardInfo();
	SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,SAMPLE_SVP_NNIE_FillForwardCtrl failed!\n");

	/*Get taskInfo and Blob mem size*/
	s32Ret = SAMPLE_SVP_NNIE_GetTaskAndBlobBufSize(&u32TotalTaskBufSize,
		&u32TmpBufSize, astBlobSize, &u32TotalSize);
	SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,SAMPLE_SVP_NNIE_GetTaskAndBlobBufSize failed!\n");

	/*Malloc mem*/
	//printf("init malloc & cached\n");
	s32Ret = SAMPLE_COMM_SVP_MallocCached((HI_CHAR*)"SAMPLE_NNIE_TASK", NULL, (HI_U64*)&u64PhyAddr, (void**)&pu8VirAddr, u32TotalSize);
	SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,Malloc memory failed!\n");
	//printf("malloc %d at %llu %llu\n",u32TotalSize,pu8VirAddr,u64PhyAddr);
	memset(pu8VirAddr, 0, u32TotalSize);
	SAMPLE_COMM_SVP_FlushCache(u64PhyAddr, (void*)pu8VirAddr, u32TotalSize);

	/*fill taskinfo mem addr*/
	stTaskBuf.u32Size = u32TotalTaskBufSize;
	stTaskBuf.u64PhyAddr = u64PhyAddr;
	stTaskBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;

	/*fill Tmp mem addr*/
	stTmpBuf.u32Size = u32TmpBufSize;
	stTmpBuf.u64PhyAddr = u64PhyAddr + u32TotalTaskBufSize;
	stTmpBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr + u32TotalTaskBufSize;

	/*fill forward ctrl addr*/
	for (i = 0; i < stModel.u32NetSegNum; i++)
	{
		if (SVP_NNIE_NET_TYPE_ROI == stModel.astSeg[i].enNetType)
		{
			astForwardWithBboxCtrl[i].stTmpBuf = stTmpBuf;
			astForwardWithBboxCtrl[i].stTskBuf.u64PhyAddr = stTaskBuf.u64PhyAddr + u32Offset;
			astForwardWithBboxCtrl[i].stTskBuf.u64VirAddr = stTaskBuf.u64VirAddr + u32Offset;
			astForwardWithBboxCtrl[i].stTskBuf.u32Size = au32TaskBufSize[i];
		}
		else if (SVP_NNIE_NET_TYPE_CNN == stModel.astSeg[i].enNetType ||
			SVP_NNIE_NET_TYPE_RECURRENT == stModel.astSeg[i].enNetType)
		{
			astForwardCtrl[i].stTmpBuf = stTmpBuf;
			astForwardCtrl[i].stTskBuf.u64PhyAddr = stTaskBuf.u64PhyAddr + u32Offset;
			astForwardCtrl[i].stTskBuf.u64VirAddr = stTaskBuf.u64VirAddr + u32Offset;
			astForwardCtrl[i].stTskBuf.u32Size = au32TaskBufSize[i];
		}
		/**** add tsk ****/
		s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&astForwardCtrl[i].stTskBuf);

		u32Offset += au32TaskBufSize[i];
	}

	/*fill each blob's mem addr*/
	u64PhyAddr = u64PhyAddr + u32TotalTaskBufSize + u32TmpBufSize;
	pu8VirAddr = pu8VirAddr + u32TotalTaskBufSize + u32TmpBufSize;
	for (i = 0; i < stModel.u32NetSegNum; i++)
	{
		/*first seg has src blobs, other seg's src blobs from the output blobs of
		those segs before it or from software output results*/
		//if (0 == i)
		//{
			for (j = 0; j < stModel.astSeg[i].u16SrcNum; j++)
			{
				if (j != 0)
				{
					u64PhyAddr += astBlobSize[i].au32SrcSize[j - 1];
					pu8VirAddr += astBlobSize[i].au32SrcSize[j - 1];
				}
				astSegData[i].astSrc[j].u64PhyAddr = u64PhyAddr;
				astSegData[i].astSrc[j].u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;
			}
			u64PhyAddr += astBlobSize[i].au32SrcSize[j - 1];
			pu8VirAddr += astBlobSize[i].au32SrcSize[j - 1];
		//}

		/*fill the mem addrs of each seg's output blobs*/
		for (j = 0; j < stModel.astSeg[i].u16DstNum; j++)
		{
			if (j != 0)
			{
				u64PhyAddr += astBlobSize[i].au32DstSize[j - 1];
				pu8VirAddr += astBlobSize[i].au32DstSize[j - 1];
			}
			astSegData[i].astDst[j].u64PhyAddr = u64PhyAddr;
			astSegData[i].astDst[j].u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;
		}
		u64PhyAddr += astBlobSize[i].au32DstSize[j - 1];
		pu8VirAddr += astBlobSize[i].au32DstSize[j - 1];
	}
	return s32Ret;
}

HI_S32 NNIE_Net::SAMPLE_SVP_NNIE_FillForwardInfo(/*SAMPLE_SVP_NNIE_CFG_S * pstNnieCfg*/)
{
	HI_U32 i = 0, j = 0;
	HI_U32 u32Offset = 0;
	HI_U32 u32Num = 0;

	for (i = 0; i < stModel.u32NetSegNum; i++)
	{
		/*fill forwardCtrl info*/
		if (SVP_NNIE_NET_TYPE_ROI == stModel.astSeg[i].enNetType)
		{
			astForwardWithBboxCtrl[i].enNnieId = aenNnieCoreId[i];
			astForwardWithBboxCtrl[i].u32SrcNum = stModel.astSeg[i].u16SrcNum;
			astForwardWithBboxCtrl[i].u32DstNum = stModel.astSeg[i].u16DstNum;
			astForwardWithBboxCtrl[i].u32ProposalNum = 1;
			astForwardWithBboxCtrl[i].u32NetSegId = i;
			astForwardWithBboxCtrl[i].stTmpBuf = stTmpBuf;
			astForwardWithBboxCtrl[i].stTskBuf.u64PhyAddr = stTaskBuf.u64PhyAddr + u32Offset;
			astForwardWithBboxCtrl[i].stTskBuf.u64VirAddr = stTaskBuf.u64VirAddr + u32Offset;
			astForwardWithBboxCtrl[i].stTskBuf.u32Size = au32TaskBufSize[i];
		}
		else if (SVP_NNIE_NET_TYPE_CNN == stModel.astSeg[i].enNetType ||
			SVP_NNIE_NET_TYPE_RECURRENT == stModel.astSeg[i].enNetType)
		{
			astForwardCtrl[i].enNnieId = aenNnieCoreId[i];
			astForwardCtrl[i].u32SrcNum = stModel.astSeg[i].u16SrcNum;
			astForwardCtrl[i].u32DstNum = stModel.astSeg[i].u16DstNum;
			astForwardCtrl[i].u32NetSegId = i;
			astForwardCtrl[i].stTmpBuf = stTmpBuf;
			astForwardCtrl[i].stTskBuf.u64PhyAddr = stTaskBuf.u64PhyAddr + u32Offset;
			astForwardCtrl[i].stTskBuf.u64VirAddr = stTaskBuf.u64VirAddr + u32Offset;
			astForwardCtrl[i].stTskBuf.u32Size = au32TaskBufSize[i];
		}
		u32Offset += au32TaskBufSize[i];
		/*fill src blob info*/
		for (j = 0; j < stModel.astSeg[i].u16SrcNum; j++)
		{
			/*Recurrent blob*/
			if (SVP_BLOB_TYPE_SEQ_S32 == stModel.astSeg[i].astSrcNode[j].enType)
			{
				astSegData[i].astSrc[j].enType = stModel.astSeg[i].astSrcNode[j].enType;
				astSegData[i].astSrc[j].unShape.stSeq.u32Dim = stModel.astSeg[i].astSrcNode[j].unShape.u32Dim;
				astSegData[i].astSrc[j].u32Num = u32MaxInputNum;
				astSegData[i].astSrc[j].unShape.stSeq.u64VirAddrStep = au64StepVirAddr[i*SAMPLE_SVP_NNIE_EACH_SEG_STEP_ADDR_NUM];
			}
			else
			{
				astSegData[i].astSrc[j].enType = stModel.astSeg[i].astSrcNode[j].enType;
				astSegData[i].astSrc[j].unShape.stWhc.u32Chn = stModel.astSeg[i].astSrcNode[j].unShape.stWhc.u32Chn;
				astSegData[i].astSrc[j].unShape.stWhc.u32Height = stModel.astSeg[i].astSrcNode[j].unShape.stWhc.u32Height;
				astSegData[i].astSrc[j].unShape.stWhc.u32Width = stModel.astSeg[i].astSrcNode[j].unShape.stWhc.u32Width;
				astSegData[i].astSrc[j].u32Num = u32MaxInputNum;
			}
		}

		/*fill dst blob info*/
		if (SVP_NNIE_NET_TYPE_ROI == stModel.astSeg[i].enNetType)
		{
			u32Num = u32MaxRoiNum*u32MaxInputNum;
		}
		else
		{
			u32Num = u32MaxInputNum;
		}

		for (j = 0; j < stModel.astSeg[i].u16DstNum; j++)
		{
			if (SVP_BLOB_TYPE_SEQ_S32 == stModel.astSeg[i].astDstNode[j].enType)
			{
				astSegData[i].astDst[j].enType = stModel.astSeg[i].astDstNode[j].enType;
				astSegData[i].astDst[j].unShape.stSeq.u32Dim = stModel.astSeg[i].astDstNode[j].unShape.u32Dim;
				astSegData[i].astDst[j].u32Num = u32Num;
				astSegData[i].astDst[j].unShape.stSeq.u64VirAddrStep = au64StepVirAddr[i*SAMPLE_SVP_NNIE_EACH_SEG_STEP_ADDR_NUM + 1];
			}
			else
			{
				astSegData[i].astDst[j].enType = stModel.astSeg[i].astDstNode[j].enType;
				astSegData[i].astDst[j].unShape.stWhc.u32Chn = stModel.astSeg[i].astDstNode[j].unShape.stWhc.u32Chn;
				astSegData[i].astDst[j].unShape.stWhc.u32Height = stModel.astSeg[i].astDstNode[j].unShape.stWhc.u32Height;
				astSegData[i].astDst[j].unShape.stWhc.u32Width = stModel.astSeg[i].astDstNode[j].unShape.stWhc.u32Width;
				astSegData[i].astDst[j].u32Num = u32Num;
			}
		}
	}
	return HI_SUCCESS;
}

HI_S32 NNIE_Net::SAMPLE_SVP_NNIE_GetTaskAndBlobBufSize(/*SAMPLE_SVP_NNIE_CFG_S * pstNnieCfg,*/ HI_U32 * pu32TotalTaskBufSize, HI_U32 * pu32TmpBufSize, SAMPLE_SVP_NNIE_BLOB_SIZE_S astBlobSize[], HI_U32 * pu32TotalSize)
{
	HI_S32 s32Ret = HI_SUCCESS;
	HI_U32 i = 0, j = 0;
	HI_U32 u32TotalStep = 0;

	/*Get each seg's task buf size*/
	s32Ret = HI_MPI_SVP_NNIE_GetTskBufSize(u32MaxInputNum, u32MaxRoiNum, &stModel, au32TaskBufSize, stModel.u32NetSegNum);
	SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,HI_MPI_SVP_NNIE_GetTaskSize failed!\n");

	/*Get total task buf size*/
	*pu32TotalTaskBufSize = 0;
	for (i = 0; i < stModel.u32NetSegNum; i++)
	{
		*pu32TotalTaskBufSize += au32TaskBufSize[i];
	}

	/*Get tmp buf size*/
	*pu32TmpBufSize = stModel.u32TmpBufSize;
	*pu32TotalSize += *pu32TotalTaskBufSize + *pu32TmpBufSize;

	/*calculate Blob mem size*/
	for (i = 0; i < stModel.u32NetSegNum; i++)
	{
		if (SVP_NNIE_NET_TYPE_RECURRENT == stModel.astSeg[i].enNetType)
		{
			for (j = 0; j < astSegData[i].astSrc[0].u32Num; j++)
			{
				u32TotalStep += *((HI_S32*)(HI_UL)astSegData[i].astSrc[0].unShape.stSeq.u64VirAddrStep + j);
			}
		}
		/*the first seg's Src Blob mem size, other seg's src blobs from the output blobs of
		those segs before it or from software output results*/
		//if (i == 0)
		//{
			SAMPLE_SVP_NNIE_GetBlobMemSize(&(stModel.astSeg[i].astSrcNode[0]),
				stModel.astSeg[i].u16SrcNum, u32TotalStep, &(astSegData[i].astSrc[0]),
				SAMPLE_SVP_NNIE_ALIGN_16, pu32TotalSize, &(astBlobSize[i].au32SrcSize[0]));
		//}
		//else
		//	for (int s = 0; s < stModel.astSeg[i].u16SrcNum; s++)
		//		astSegData[i].astSrc[s].u32Stride = this->Cal_stride(astSegData[i].astSrc[s].enType,
		//			astSegData[i].astSrc[s].unShape.stWhc.u32Width);



		/*Get each seg's Dst Blob mem size*/
		SAMPLE_SVP_NNIE_GetBlobMemSize(&(stModel.astSeg[i].astDstNode[0]),
			stModel.astSeg[i].u16DstNum, u32TotalStep, &(astSegData[i].astDst[0]),
			SAMPLE_SVP_NNIE_ALIGN_16, pu32TotalSize, &(astBlobSize[i].au32DstSize[0]));
	}
	return s32Ret;
}

HI_S32 NNIE_Net::SAMPLE_SVP_NNIE_Forward(SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S * pstInputDataIdx, SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S * pstProcSegIdx, HI_BOOL bInstant)
{
	HI_S32 s32Ret = HI_SUCCESS;
	HI_U32 i = 0, j = 0;
	HI_BOOL bFinish = HI_FALSE;
	SVP_NNIE_HANDLE hSvpNnieHandle = 0;
	HI_U32 u32TotalStepNum = 0;
	SAMPLE_COMM_SVP_FlushCache(astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
		SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr),
		astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);
	for (i = 0; i < astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
	{
		if (SVP_BLOB_TYPE_SEQ_S32 == astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
		{
			for (j = 0; j < astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
			{
				u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32, astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep) + j);
			}
			SAMPLE_COMM_SVP_FlushCache(astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
				SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
				u32TotalStepNum*astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
		}
		else
		{
			SAMPLE_COMM_SVP_FlushCache(astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
				SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
				astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
				astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
				astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
				astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
		}
	}
	/*set input blob according to node name*/
	if (pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx)
	{
		for (i = 0; i < stModel.astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++)
		{
			for (j = 0; j < stModel.astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++)
			{
				if (0 == strncmp(stModel.astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
					stModel.astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
					SVP_NNIE_NODE_NAME_LEN))
				{
					astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] = astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
					break;
				}
			}
			SAMPLE_SVP_CHECK_EXPR_RET((j == stModel.astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
				HI_FAILURE, SAMPLE_SVP_ERR_LEVEL_ERROR, "Error,can't find %d-th seg's %d-th src blob!\n", pstProcSegIdx->u32SegIdx, i);
		}
	}

	/*NNIE_Forward*/
	s32Ret = HI_MPI_SVP_NNIE_Forward(&hSvpNnieHandle,
		astSegData[pstProcSegIdx->u32SegIdx].astSrc,
		&stModel, astSegData[pstProcSegIdx->u32SegIdx].astDst,
		&astForwardCtrl[pstProcSegIdx->u32SegIdx], bInstant);
	SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret, s32Ret, SAMPLE_SVP_ERR_LEVEL_ERROR,
		"Error,HI_MPI_SVP_NNIE_Forward failed!\n");

	if (bInstant)
	{
		/*Wait NNIE finish*/
		while (HI_ERR_SVP_NNIE_QUERY_TIMEOUT == (s32Ret = HI_MPI_SVP_NNIE_Query(astForwardCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
			hSvpNnieHandle, &bFinish, HI_TRUE)))
		{
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
			usleep(100);
#endif
			SAMPLE_SVP_TRACE(SAMPLE_SVP_ERR_LEVEL_INFO,
				"HI_MPI_SVP_NNIE_Query Query timeout!\n");
		}
	}
#if 1
	u32TotalStepNum = 0;
	for (i = 0; i < astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
	{
		if (SVP_BLOB_TYPE_SEQ_S32 == astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
		{
			for (j = 0; j < astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
			{
				u32TotalStepNum += *(SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_U32, astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep) + j);
			}
			SAMPLE_COMM_SVP_FlushCache(astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
				SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
				u32TotalStepNum*astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
		}
		else
		{
			SAMPLE_COMM_SVP_FlushCache(astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
				SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr),
				astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
				astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
				astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
				astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
		}
	}
#endif
	return s32Ret;
}

HI_S32 NNIE_Net::SVP_FillSrcData_Mat(SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S * stInputDataIdx, const cv::Mat src)
{
	if (src.empty())
		return HI_FAILURE;
	HI_U32 u32Height = astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].unShape.stWhc.u32Height;
	HI_U32 u32Width = astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].unShape.stWhc.u32Width;
	HI_U32 u32Chn = astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].unShape.stWhc.u32Chn;
	HI_U32 u32Stride = astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].u32Stride;

	cv::Mat resize_src;
	cv::resize(src, resize_src, cv::Size(u32Width, u32Height));
	if (resize_src.channels() > u32Chn)
		cv::cvtColor(resize_src, resize_src, CV_GRAY2BGR);
	else if (resize_src.channels() < u32Chn)
		cv::cvtColor(resize_src, resize_src, CV_BGR2GRAY);
	HI_U8* pu8PicAddr = (HI_U8*)((HI_UL*)astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].u64VirAddr);
	std::vector<cv::Mat> bgrs;
	cv::split(resize_src, bgrs);
	for (int cc = 0; cc < u32Chn; cc++) {

		for (int h = 0; h < u32Height; h++) {
			uchar *data0 = bgrs[cc].ptr<uchar>(h);
			memcpy(pu8PicAddr, data0, u32Width);
			pu8PicAddr += u32Stride;
		}
	}
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	HI_MPI_SYS_MmzFlushCache(astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].u64PhyAddr,
		SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].u64VirAddr),
		astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].u32Num*u32Chn*u32Height*u32Stride);
#else
	SAMPLE_COMM_SVP_FlushCache(astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].u64PhyAddr,
		SAMPLE_SVP_NNIE_CONVERT_64BIT_ADDR(HI_VOID, astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].u64VirAddr),
		astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].u32Num*
		astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].unShape.stWhc.u32Chn*
		astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].unShape.stWhc.u32Height*
		astSegData[stInputDataIdx->u32SegIdx].astSrc[stInputDataIdx->u32NodeIdx].u32Stride);
#endif
	return HI_SUCCESS;
}

HI_U32 NNIE_Net::Cal_stride(SVP_BLOB_TYPE_E enType, HI_U32 width)
{
	HI_U8 u32Size;
	if (SVP_BLOB_TYPE_S32 == enType || SVP_BLOB_TYPE_VEC_S32 == enType ||
		SVP_BLOB_TYPE_SEQ_S32 == enType)
	{
		u32Size = sizeof(HI_U32);
	}
	else
	{
		u32Size = sizeof(HI_U8);
	}
	return SAMPLE_SVP_NNIE_ALIGN16(width*u32Size);
}

