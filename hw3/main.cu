#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include "file.h"

//
#define PAGESIZE 32
//
#define PHYSICAL_MEM_SIZE 32768
//
#define STORAGE_SIZE 131072

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;

//
__device__ __managed__ int PAGE_ENTRIES = 0;
//count the pagefault times
__device__ __managed__ int PAGEFAULT = 0;

//secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE];

//data input & output
__device__ __managed__ uchar result[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

//page table
extern __shared__ u32 pt[];

/******BLABLABLA~~****
**********************/

__device__ uchar Gread(uchar *buffer, u32 addr) {
	u32 frame_num = addr/PAGESIZE;
	u32 offset = addr%PAGESIZE;

	/*addr = paging(buffer, frame_num, offset);*/
	return buffer[addr];
}

__device__ void Gwrite(uchar *buffer, u32 addr, uchar value) {
	u32 frame_num = addr/PAGESIZE;
	u32 offset = addr%PAGESIZE;

	/*addr = paging(buffer, frame_num, offset);*/
	buffer[addr] = value;
}

__device__ void snapshot(uchar *result, uchar *buffer, int offset, int input_size) {
	for(int i = 0; i < input_size; i++) result[i] = Gread(buffer, i + offset);
}

__global__ void mykernel(int input_size) {
	//take shared memory as physical memory
	__shared__ uchar data[PHYSICAL_MEM_SIZE];
	//get page table entries
	int pt_entries = PHYSICAL_MEM_SIZE/PAGESIZE;

	//B4 1st Gwrite or Gread
	/*init_pageTable(pt_entries);*/

	//####Gwrite/Gread code section start####
	for(int i = 0; i < input_size; i++) Gwrite(data, i, input[i]);
	for(int i = input_size-1; i >= input_size-10; i--) int value = Gread(data, i);

	//the last line of Gwrite/Gread code section should be snapshot()
	snapshot(result, data, 0, input_size);
	//####Gwrite/Gread code section end####

	printf("pagefault times = %d\n", PAGEFAULT);
}

int main() {
	int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);

	printf("%d\n", input_size);
	/*mykernel<<<1, 1, 16384>>>(input_size);*/
	cudaSetDevice(3);
	cudaDeviceSynchronize();
	cudaDeviceReset();

	write_binaryFIle(OUTFILE, result, input_size);

	return 0;
}
