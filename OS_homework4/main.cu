#include <stdio.h>
#include <inttypes.h>

#define STORAGE_SIZE 1085440
#define MAX_FILE_SIZE 1048576

#define BLOCK_SIZE 1024
#define INODE_START 2
#define INODE_SIZE 28
#define INODE_COUNT 1024
#define INODE_LOC(i) (INODE_START + i*INODE_SIZE)
#define DATA_START 36864
#define ERROR 65535

#define DATAFILE "data.bin"
#define OUTFILE "snapshot.bin"
#define G_WRITE 0
#define G_READ 1
#define WRITE_SUCCESS 0
#define WRITE_ERROR 1
#define READ_SUCCESS 0
#define READ_ERROR 1

#define NAME_LENGTH 21

#define LS_D 0
#define LS_S 1
#define RM 2

#define TIME_LOC DATA_START-2
#define TIME (read2bytes(TIME_LOC))

typedef unsigned char uchar;
typedef uint32_t u32;

__device__ __managed__ uchar *volume;

int load_binaryFile(const char *filename, uchar *a, int max_size);
void write_binaryFIle(const char *filename, uchar *a, int size);
__device__ u32 open(const char *name, uchar mode);
__device__ uchar write(uchar *input, int n, uchar fp);
__device__ uchar read(uchar *output, int n, uchar fp);
__device__ void gsys(uchar arg, const char* file=NULL);
__device__ void init_volume();

__device__ int my_strcmp(uchar *a, const char *b) {
	for(int i = 0; i < NAME_LENGTH; i++) {
		if(a[i] == 0 && b[i] == 0) return 0;
		else if(a[i] != b[i]) return 1;
	}
	return 1;
}
__device__ void my_strcpy(uchar *d, const char *s) {
	int i;
	for(i = 0; i < NAME_LENGTH-1; i++) {
		d[i] = s[i];
		if(s[i] == 0) return;
	}
	d[i] = 0;
}
__device__ int read2bytes(int i) {
	return (volume[i]<<8) + volume[i+1];
}
__device__ void write2bytes(int num, int i) {
	volume[i+1] = num;
	volume[i] = (num>>8);
}
__device__ int find_room() {
	int i = read2bytes(0);
	int cur = INODE_LOC(i);
	int j = read2bytes(cur+1);
	cur = INODE_LOC(j);
	int k = read2bytes(cur+1);
	write2bytes(k, 0);
	return j;
}
__device__ void free_room(int v) {
	int i = read2bytes(0);
	int cur = INODE_LOC(v);
	volume[cur] = 0; //set to empty
	write2bytes(v, 0);
	write2bytes(i, cur+1);
}

__global__ void mykernel(uchar *input, uchar *output) {
	init_volume();
	//####kernel start####
	u32 fpa = open("a.txt\0", G_WRITE);
	write(input, 30, fpa);
	u32 fpb = open("b.txt\0", G_WRITE);
	write(input, 10, fpa);
	gsys(LS_S);
	gsys(RM, "a.txt\0");
	gsys(LS_D);
	read(output, 5, fpa);
	//####kernel end####
}
int main() {
	cudaSetDevice(3);
	cudaMallocManaged(&volume, STORAGE_SIZE);

	uchar *input, *output;
	cudaMallocManaged(&input, MAX_FILE_SIZE);
	cudaMallocManaged(&output, MAX_FILE_SIZE);
	for(int i = 0; i < MAX_FILE_SIZE; i++) output[i] = 0;

	load_binaryFile(DATAFILE, input, MAX_FILE_SIZE);

	mykernel<<<1, 1>>>(input, output);
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
__device__ void init_volume() {
	write2bytes(0, TIME_LOC); //time init
	for(int i = 0; i < INODE_COUNT; i++) {
		int cur = INODE_LOC(i);
		volume[cur] = 0; //set empty
		write2bytes(i+1, cur+1); //point to i+1
	}
}
__device__ u32 open(const char *name, uchar mode) {
	for(int i = 0; i < INODE_COUNT; i++) {
		int cur = INODE_LOC(i);
		if(my_strcmp(volume+cur+7, name) == 0) {
			write2bytes(0, cur+1); //set fp to 0
			return i;
		}
	}
	if(mode == G_WRITE) { //create
		int i = find_room();
		if(i == -1) return ERROR;
		int cur = INODE_LOC(i);
		volume[cur] = 1; //set not empty
		write2bytes(0, cur+1); //set fp to 0
		write2bytes(0, cur+3); //set size to 0
		int time = TIME;
		write2bytes(time, cur+5); //set timestamp
		write2bytes(time+1, TIME_LOC); //increase time
		my_strcpy(volume+cur+7, name); //set name
		return i;
	}
	else return ERROR;
}

__device__ uchar write(uchar *input, int n, uchar fp) {
	if(fp == ERROR) return WRITE_ERROR;
	if(n > BLOCK_SIZE) return WRITE_ERROR;
	int cur = INODE_LOC(fp);
	int cur_block = DATA_START + fp*BLOCK_SIZE;
	write2bytes(n, cur+3); //change file size
	int time = TIME;
	write2bytes(time, cur+5); //set time stamp
	write2bytes(time+1, TIME_LOC); //increase time
	for(int i = 0; i < n; i++) {
		volume[cur_block+i] = input[i];
	}
	return WRITE_SUCCESS;
}
__device__ uchar read(uchar *output, int n, uchar fp) {
	if(fp == ERROR) return READ_ERROR;
	if(n > BLOCK_SIZE) return READ_ERROR;
	int cur_block = DATA_START + fp*BLOCK_SIZE;
	for(int i = 0; i < n; i++) {
		output[i] = volume[cur_block+i];
	}
	return READ_SUCCESS;
}
__device__ void gsys(uchar arg, const char *file) {
	if(arg == LS_S) {
		printf("===sorted by file size===\n");
		int a[INODE_COUNT];
		int n = 0;
		for(int i = 0; i < INODE_COUNT; i++) {
			int cur = INODE_LOC(i);
			if(volume[cur]) a[n++] = cur; //inode not empty
		}
		for(int i = 0; i < n; i++) {
			for(int j = i+1; j < n; j++) {
				if(read2bytes(a[i]+3) > read2bytes(a[j]+3)) {
					int tmp = a[i];
					a[i] = a[j];
					a[j] = tmp;
				}
			}
		}
		for(int i = 0; i < n; i++) printf("%s %d\n", volume+a[i]+7, read2bytes(a[i]+3));
	}
	else if(arg == LS_D) {
		printf("===sorted by modified time===\n");
		int a[INODE_COUNT];
		int n = 0;
		for(int i = 0; i < INODE_COUNT; i++) {
			int cur = INODE_LOC(i);
			if(volume[cur]) a[n++] = cur; //inode not empty
		}
		for(int i = 0; i < n; i++) {
			for(int j = i+1; j < n; j++) {
				if(read2bytes(a[i]+5) < read2bytes(a[j]+5)) {
					int tmp = a[i];
					a[i] = a[j];
					a[j] = tmp;
				}
			}
		}
		for(int i = 0; i < n; i++) printf("%s\n", volume+a[i]+7);
	}
	else if(arg == RM) {
		int i;
		for(i = 0; i < INODE_COUNT; i++) {
			int cur = INODE_LOC(i);
			if(volume[cur]) { //not empty
				if(my_strcmp(volume+cur+7, file) == 0) break;
			}
		}
		if(i == INODE_COUNT) printf("No such file %s!\n", file);
		else {
			free_room(i);
		}
	}
}
