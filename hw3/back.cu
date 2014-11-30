#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

//
#define PAGESIZE 32
//
#define PHYSICAL_MEM_SIZE 32768
//
#define STORAGE_SIZE 131072

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

#define MASK 32767
#define TIME_MAX 4294967295
typedef unsigned char uchar;
typedef uint32_t u32;

//
__device__ __managed__ int PAGE_ENTRIES = PHYSICAL_MEM_SIZE/PAGESIZE;
//count the pagefault times
__device__ __managed__ int PAGEFAULT = 0;

//secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE];

//data input & output
__device__ __managed__ uchar result[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

//page table
extern __shared__ u32 pt[];

/******BLABLABLA~~****/
int load_binaryFile(const char *filename, uchar *a, int max_size) {
	FILE *fp = fopen(filename, "rb");
	int i = 0;
	while(!feof(fp) && i < max_size) {
		fread(a+i, sizeof(uchar), 1, fp);
		i++;
	}
	return i;
}

void write_binaryFIle(const char *filename, uchar *a, int size) {
	FILE *fp = fopen(filename, "wb+");
	fwrite(a, sizeof(uchar), size, fp);
}
__device__ void init_pageTable(int pt_entries) {
	for(int i = 0; i < pt_entries; i++) {
		pt[i] = (i<<15); //若還沒做Gwrite就做Gread可能會得到無意義的值，但這是使用者的錯誤，not my business.
	}
}
/*********************/

__global__ void mykernel(int input_size) {
	PAGEFAULT = 999;
	__shared__ uchar data[PHYSICAL_MEM_SIZE];
	//get page table entries
	int pt_entries = PHYSICAL_MEM_SIZE/PAGESIZE;
	//B4 1st Gwrite or Gread
	init_pageTable(pt_entries);
	data[1] = 1;
	//####Gwrite/Gread code section start####
	/*for(int i = 0; i < input_size; i++) Gwrite(data, i, input[i]);*/
	/*for(int i = input_size-1; i >= input_size-10; i--) int value = Gread(data, i);*/

	//the last line of Gwrite/Gread code section should be snapshot()
	/*snapshot(result, data, 0, input_size);*/
	//####Gwrite/Gread code section end####
}

int main() {
	int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);

	cudaSetDevice(3);
	mykernel<<<1, 1, 16384>>>(input_size);
	cudaDeviceSynchronize();
	cudaDeviceReset();

	printf("pagefault times = %d\n", PAGEFAULT);
	write_binaryFIle(OUTFILE, result, input_size);

	return 0;
}
