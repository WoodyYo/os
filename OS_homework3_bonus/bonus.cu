#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#define PAGESIZE 32

#define PHYSICAL_MEM_SIZE 32768

#define STORAGE_SIZE 131072

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"

#define F_MASK 32767
#define P_MASK 4096
#define TIME_MAX 4294967295
#define MEMORY_SEGMENT 32768

#define __LOCK(); for(int j = 0; j < 4; j++) {if(threadIdx.x == j) {
#define __UNLOCK(); }__syncthreads(); }
#define __GET_BASE() j*MEMORY_SEGMENT
//C's macro is soooo bloody ugly comparing to ruby & lisp...

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
__device__ __managed__ u32 latest_time[1024];

__device__ __managed__ u32 cur_time;

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

__device__ u32 lru() {
	/****
	  實作queue來解決lru並無法解決效能瓶頸，因為最大的問題卡在find的O(n)
	  要改善find的效能，應實作binary search tree，but...
	 ***/
	u32 min = TIME_MAX;
	int victim_index = 0;
	for(int i = 0; i < PAGE_ENTRIES; i++) {
		if(latest_time[i] == 0) return i;
		else {
			u32 cur_id = pt[i]>>27;
			if(cur_id == threadIdx.x && latest_time[i] < min) {
				min = latest_time[i];
				victim_index = i;
			}
		}
	}
	return victim_index;
}
__device__ int find(u32 p) {
	for(int i = 0; i < PAGE_ENTRIES; i++) {
		u32 cur_p_id = (pt[i]>>15);
		if(cur_p_id == (p|(threadIdx.x<<15))) {
			if(latest_time[i] == 0) return -1;
			else return i;
		}
	}
	return -1;
}
__device__ u32 paging(uchar *data, u32 p, u32 offset) {
	if(cur_time < TIME_MAX) cur_time++;
	int p_index = find(p); //should only return the page that is of same id
	if(p_index == -1) {  //page fault!!
		PAGEFAULT++;
		u32 victim_index = lru(); //should only return the page that is of same id, OR that is not used, since I can't see another thread's data[]
		u32 frame = pt[victim_index]&F_MASK;
		u32 victim_p = (pt[victim_index] >> 15)&P_MASK;
		for(int i = 0; i < 32; i++) {
			storage[threadIdx.x*MEMORY_SEGMENT+victim_p*32+i] = data[frame+i];
			data[frame+i] = storage[threadIdx.x*MEMORY_SEGMENT+p*32+i];
		}
		pt[victim_index] = ((threadIdx.x<<27)|(p<<15)|frame);
		latest_time[victim_index] = cur_time;
		return frame + offset;
	}
	else {
		latest_time[p_index] = cur_time;
		return (pt[p_index]&F_MASK) + offset;
	}
}
__device__ void init_pageTable(int pt_entries) {
	cur_time = 0;
	for(int i = 0; i < PAGE_ENTRIES; i++) {
		pt[i] = i*32;
		latest_time[i] = 0;
	}
}
/*********************/

__device__ uchar Gread(uchar *data, u32 addr) {
	u32 p = addr/PAGESIZE;
	u32 offset = addr%PAGESIZE;

	addr = paging(data, p, offset);
	return data[addr];
}

__device__ void Gwrite(uchar *data, u32 addr, uchar value) {
	u32 p = addr/PAGESIZE;
	u32 offset = addr%PAGESIZE;

	addr = paging(data, p, offset);
	data[addr] = value;
}

__device__ void snapshot(uchar *result, uchar *data, int offset, int input_size) {
	for(int i = 0; i < input_size; i++) {
		result[i] = Gread(data, i + offset);
	}
}

__global__ void mykernel(int input_size) {
	__shared__ uchar data[PHYSICAL_MEM_SIZE/4];
	//get page table entries
	int pt_entries = PHYSICAL_MEM_SIZE/PAGESIZE;

	printf("my id = %d\n", threadIdx.x);
	//B4 1st Gwrite or Gread
	if(threadIdx.x == 0) init_pageTable(pt_entries);

	//####Gwrite/Gread code section start####
	__LOCK();
	for(int i = 0; i < input_size; i++) Gwrite(data, i+__GET_BASE(), input[i]);
	__UNLOCK();
	for(int i = input_size-1; i >= input_size-10; i--) {
		__LOCK();
		int value = Gread(data, i+__GET_BASE());
		__UNLOCK();
	}

	//the last line of Gwrite/Gread code section should be snapshot()
	__LOCK();
	snapshot(result+__GET_BASE(), data, __GET_BASE(), input_size);
	__UNLOCK();
	//####Gwrite/Gread code section end####
}

int main() {
	int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);

	cudaSetDevice(3);
	mykernel<<<1, 4, 16384>>>(input_size/4);
	cudaDeviceSynchronize();
	cudaDeviceReset();

	printf("pagefault times = %d\n", PAGEFAULT);
	write_binaryFIle(OUTFILE, result, input_size);

	return 0;
}
