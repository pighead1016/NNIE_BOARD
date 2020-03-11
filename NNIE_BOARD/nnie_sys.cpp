#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include "hi_comm_svp.h"
#include <sys/stat.h>
#include <sys/types.h>
#include "nnie_sys.h"

#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
#include <unistd.h>
#include <sys/mman.h>
#include "hi_common.h"
#include "hi_comm_sys.h"
#include "mpi_sys.h"
#endif

#include "mpi_nnie.h"
#include "hi_type.h"
#include "hi_nnie.h"
#if 1
/*
*malloc
*/
HI_U32 SvpSampleAlign(HI_U32 u32Size, HI_U32 u32AlignNum)
{
	return (u32Size + (u32AlignNum - u32Size%u32AlignNum) % u32AlignNum);
}
/*
* Free mem,depend on different environment
*/
HI_VOID SvpSampleFree(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr)
{
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP

	if ((0 != u64PhyAddr) && (NULL != pvVirAddr))
	{
		(HI_VOID)HI_MPI_SYS_MmzFree(u64PhyAddr, pvVirAddr);
	}
#else
	if (NULL != pvVirAddr)
	{
		free(pvVirAddr);
	}
#endif
}

HI_VOID SvpSampleMemFree(SVP_MEM_INFO_S *pstMem)
{
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP

	if ((0 != pstMem->u64PhyAddr) && (0 != pstMem->u64VirAddr))
	{
		(HI_VOID)HI_MPI_SYS_MmzFree(pstMem->u64PhyAddr, (HI_VOID*)(HI_UL)pstMem->u64VirAddr);
	}
#else
	if (0 != pstMem->u64VirAddr)
	{
		free((void*)pstMem->u64VirAddr);
		pstMem->u64VirAddr = 0;
	}
#endif
}

/*
*Align
*/
HI_U32 SAMPLE_COMM_SVP_Align(HI_U32 u32Size, HI_U16 u16Align)
{
	HI_U32 u32Stride = u32Size + (u16Align - u32Size%u16Align) % u16Align;
	return u32Stride;
}


/*
*Malloc memory
*/

HI_S32 SvpSampleMallocMem(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U32 u32Size, SVP_MEM_INFO_S *pstMem)
{
	HI_S32 s32Ret = HI_SUCCESS;

#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	s32Ret = HI_MPI_SYS_MmzAlloc(&pstMem->u64PhyAddr, (void**)&pstMem->u64VirAddr, pchMmb, pchZone, u32Size);
	pstMem->u32Size = u32Size;
#else
	pstMem->u64VirAddr = (HI_U64)malloc(u32Size);
	if (0 == pstMem->u64VirAddr)
	{
		s32Ret = HI_FAILURE;
		pstMem->u64PhyAddr = 0;
		pstMem->u32Size = 0;
	}
	else
	{
		pstMem->u64PhyAddr = pstMem->u64VirAddr;
		pstMem->u32Size = u32Size;
	}
#endif
	return s32Ret;
}

HI_S32 SvpSampleMallocMemCached(HI_CHAR *pchMmb, HI_CHAR *pchZone, HI_U32 u32Size, SVP_MEM_INFO_S *pstMem)
{
	HI_S32 s32Ret = HI_SUCCESS;
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	s32Ret = HI_MPI_SYS_MmzAlloc_Cached(&pstMem->u64PhyAddr, (HI_VOID**)&pstMem->u64VirAddr, pchMmb, pchZone, u32Size);
	pstMem->u32Size = u32Size;
#else
	pstMem->u64VirAddr = (HI_U64)malloc(u32Size);
	if (0 == pstMem->u64VirAddr)
	{
		s32Ret = HI_FAILURE;
		pstMem->u64PhyAddr = 0;
		pstMem->u32Size = 0;
	}
	else
	{
		pstMem->u64PhyAddr = pstMem->u64VirAddr;
		pstMem->u32Size = u32Size;
	}
#endif
	return s32Ret;
}
HI_S32 SAMPLE_COMM_SVP_MallocMem(HI_CHAR *pszMmb, HI_CHAR *pszZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size)
{
	HI_S32 s32Ret = HI_SUCCESS;
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	s32Ret = HI_MPI_SYS_MmzAlloc(pu64PhyAddr, ppvVirAddr, pszMmb, pszZone, u32Size);
#else
	*ppvVirAddr = (HI_VOID *)(HI_U64)malloc(u32Size);
	if (HI_NULL == *ppvVirAddr)
	{
		s32Ret = HI_FAILURE;
	}
	else
	{
		*pu64PhyAddr = (HI_U64)*ppvVirAddr;
		memset(*ppvVirAddr, 0, u32Size);
	}

#endif
	return s32Ret;
}
/*
* Flush cache, if u32PhyAddr==0£¬that means flush all cache
*/
HI_S32 SvpSampleFlushMemCache(SVP_MEM_INFO_S *pstMem)
{
	HI_S32 s32Ret = HI_SUCCESS;
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	s32Ret = HI_MPI_SYS_MmzFlushCache(pstMem->u64PhyAddr, (HI_VOID*)pstMem->u64VirAddr, pstMem->u32Size);
#endif
	return s32Ret;
}
HI_S32 SAMPLE_COMM_SVP_FlushCache(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr, HI_U32 u32Size)
{
	HI_S32 s32Ret = HI_SUCCESS;
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	s32Ret = HI_MPI_SYS_MmzFlushCache(u64PhyAddr, pvVirAddr,u32Size);
#endif
	return s32Ret;
}
/*
*Malloc memory with cached
*/
HI_S32 SAMPLE_COMM_SVP_MallocCached(HI_CHAR *pszMmb, HI_CHAR *pszZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size)
{
	HI_S32 s32Ret = HI_SUCCESS;

#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	s32Ret = HI_MPI_SYS_MmzAlloc_Cached(pu64PhyAddr, ppvVirAddr, pszMmb, pszZone, u32Size);
#else
	*ppvVirAddr = (HI_VOID *)(HI_U64)malloc(u32Size);
	if (HI_NULL == *ppvVirAddr)
	{
		s32Ret = HI_FAILURE;
	}
	else
	{
		*pu64PhyAddr = (HI_U64)*ppvVirAddr;
		memset(*ppvVirAddr, 0, u32Size);
	}
#endif
//printf("malloc cached return %d\n",s32Ret);
	return s32Ret;
}
#else
/*
*Fulsh cached
*/
HI_S32 SAMPLE_COMM_SVP_FlushCache(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr, HI_U32 u32Size)
{
	HI_S32 s32Ret = HI_SUCCESS;
	s32Ret = HI_MPI_SYS_MmzFlushCache(u64PhyAddr, pvVirAddr,u32Size);
	return s32Ret;
}

HI_S32 SAMPLE_COMM_SVP_MallocMem(HI_CHAR *pszMmb, HI_CHAR *pszZone, HI_U64 *pu64PhyAddr, HI_VOID **ppvVirAddr, HI_U32 u32Size)
{
	HI_S32 s32Ret = HI_SUCCESS;
	s32Ret = HI_MPI_SYS_MmzAlloc(pu64PhyAddr, ppvVirAddr, pszMmb, pszZone, u32Size);
	return s32Ret;
}
/*
* Free mem,depend on different environment
*/
HI_VOID SvpSampleFree(HI_U64 u64PhyAddr, HI_VOID *pvVirAddr)
{
#if ((defined __arm__) || (defined __aarch64__)) && defined HISI_CHIP
	if ((0 != u64PhyAddr) && (NULL != pvVirAddr))
	{
		(HI_VOID)HI_MPI_SYS_MmzFree(u64PhyAddr, pvVirAddr);
	}
#else
	if (NULL != pvVirAddr)
	{
		free(pvVirAddr);
	}
#endif
}
#endif
