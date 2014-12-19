#include <stdarg.h>

#define STORAGE_SIZE 1085440
#define MAX_FILE_SIZE 1048576
#define OFFSET 36864

#define DATAFILE "data.bin"
#define OUTFILE "snapshot.bon"
#define G_WRITE 0
#define G_READ 1
#define WRITE_SUCCESS 0
#define WRITE_ERROR 1
#define READ_SUCCESS 0
#define READ_ERROR 1

#define LS_D 0
#define LS_S 1
#define RM 2

typedef unsigned char uchar;
typedef uint32_t u32;

__device__ __managed__ uchar *volume;

__device__ int load_binaryFile(const char *filename, uchar *a, int max_size);
__device__ void write_binaryFIle(const char *filename, uchar *a, int size);
__device__ u32 open(const char *name, uchar mode);
__device__ uchar write(uchar *input, int n, uchar fp);
__device__ uchar read(uchar *output, int n, uchar fp);
__device__ void gsys(uchar arg, ...);
void init_volume();

__global__ void mykernel(uchar *input, uchar *output) {
	//####kernel start####

	//####kernel end####
}
int main() {
	cudaSetDevice(3);
	cudaMallocManaged(&volume, STORAGE_SIZE);
	init_volume();

	uchar *input, *output;
	cudaMallocManaged(&input, MAX_FILE_SIZE);
	cudaMallocManaged(&output, MAX_FILE_SIZE);
	for(int i = 0; i < MAX_FILE_SIZE; i++) output[i] = 0;

	load_binaryFile(DATAFILE, input, MAX_FILE_SIZE);

	mykernel<<1, 1>>(input, output);
	cudaDeviceSynchronize();
	write_binaryFIle(OUTFILE, output, MAX_FILE_SIZE);
	cudaDeviceReset();

	return 0;
}

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
void init_volume() {
	for(int i = 0; i < STORAGE_SIZE; i++) {
		volume[i] = 0;
	}
}
__device__ void gsys(uchar arg, ...) {
	if(arg == LS_S) {

	}
	else if(arg == LS_D) {

	}
	else {
		va_list va;
		va_start(va, arg);
		char *path = va_arg(va, char*);
		if(arg == RM) {

		}
		va_end(va);
	}
}
