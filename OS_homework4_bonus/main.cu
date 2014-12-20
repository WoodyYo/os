#include <stdio.h>
#include <inttypes.h>

#define STORAGE_SIZE 1085440
#define MAX_FILE_SIZE 1048576

#define BLOCK_SIZE 1024
#define INODE_START 2
#define INODE_SIZE 29
#define INODE_COUNT 1024
#define INODE_LOC(i) (INODE_START + i*INODE_SIZE)
#define BLOCK_LOC(i) (DATA_START + i*BLOCK_SIZE)
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
#define PWD 3
#define CD 4
#define CD_P 5

#define TIME_LOC DATA_START-2
#define TIME (read2bytes(TIME_LOC))
#define INC_TIME (write2bytes(TIME+1, TIME_LOC))

#define CUR_DIR_LOC DATA_START-4
#define CUR_DIR (read2bytes(CUR_DIR_LOC))
#define SET_CUR_DIR(i) (write2bytes(i, CUR_DIR_LOC))
#define MAX_CAPACITY 50

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
	//put the file in cur_dir
	int curi = INODE_LOC(CUR_DIR);
	int n = read2bytes(curi+3);
	int curb = BLOCK_LOC(CUR_DIR);
	write2bytes(j, curb + n*2);
	write2bytes(n+1, curi+3);

	return j;
}
__device__ void free_room(int v) {
	int i = read2bytes(0);
	int cur = INODE_LOC(v);
	isempty(i, 0); //set to empty
	write2bytes(v, 0);
	write2bytes(i, cur+1);
}
__device__ void my_pwd(int i) {
	if(i == 0) {
		printf("/");
		return;
	}
	int father = child(i, 0);
	my_pwd(father);
	printf("%s", name(i));
}
__device__ int isempty(int i, int v=-1) {
	int cur = INODE_LOC(i);
	if(v == -1) return volume[cur];
	else volume[cur] = v;
	return 0;
}
__device__ int next_empty(int i, int v=-1) {
	int cur = INODE_LOC(i);
	if(v == -1) return read2bytes(cur+1);
	else write2bytes(v, cur+1);
	return 0;
}
__device__ int size(int i, int v=-1) {
	int cur = INODE_LOC(i);
	if(v == -1) return read2bytes(cur+3);
	else write2bytes(v, cur+3);
	return 0;
}
__device__ int timestamp(int i, int v=-1) {
	int cur = INODE_LOC(i);
	if(v == -1) return read2bytes(cur+5);
	else write2bytes(v, cur+5);
	return 0;
}
__device__ uchar* name(int i, const char *v=NULL) {
	int cur = INODE_LOC(i);
	if(v == NULL) return volume+cur+7;
	else my_strcpy(volume+cur+7, v);
	return NULL;
}
__device__ int isdir(int i, int v=-1) {
	int cur = INODE_LOC(i);
	if(v == -1) return volume[cur+28];
	else volume[cur+28] = v;
	return 0;
}
__device__ int child(int i, int num, int v=-1) {
	int cur = BLOCK_LOC(i);
	if(v == -1) {
		int _father = read2bytes(cur + num*2);
		return (_father & 1023);
	}
	else {
		int memo = (volume[cur + num*2] & 64512); //1111110000000000
		write2bytes(v, cur + num*2);
		volume[cur + num*2] |= memo;
		return 0;
	}
}
__device__ int next_child(int i, int num, int v=-1) {
	int cur = BLOCK_LOC(i);
	if(v == -1) return (volume[cur + num*2]>>10);
	else {
		volume[cur + num*2] &= 1023;
		volume[cur + num*2] |= (v<<10);
		return 0;
	}
}

__global__ void mykernel(uchar *input, uchar *output) {
	init_volume();
	//####kernel start####
	u32 fpa = open("a.txt\0", G_WRITE);
	write(input, 30, fpa);
	write(input, 10, fpa);
	gsys(LS_S);
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
	// create root dir
	isempty(0, 1); //not empty
	size(0, 0); //capacity
	timestamp(0, 0); //timestamp
	name(0, "/\0");
	isdir(0, 1); //is a dir
	SET_CUR_DIR(0); //set cur_dir to root

	INC_TIME; //time init

	for(int i = 1; i < INODE_COUNT; i++) {
		isempty(0, 0); //set empty
		next_empty(i, i+1); //point to i+1
	}
}
__device__ u32 open(const char *file, uchar mode) {
	for(int i = 0; i < INODE_COUNT; i++) {
		if(my_strcmp(name(i), file) == 0) {
			next_empty(i, 0); //set fp to 0
			return i;
		}
	}
	if(mode == G_WRITE) { //create
		int i = find_room();
		if(i == -1) return ERROR;
		isempty(i, 1); //set not empty
		next_empty(i, 0); //set fp to 0
		size(i, 0); //set size to 0
		timestamp(i, TIME); //set timestamp
		INC_TIME; //increase time
		name(i, file); //set name
		return i;
	}
	else return ERROR;
}

__device__ uchar write(uchar *input, int n, uchar fp) {
	if(fp == ERROR) return WRITE_ERROR;
	if(n > BLOCK_SIZE) return WRITE_ERROR;
	int curb = BLOCK_LOC(fp);
	size(fp, n); //change file size
	timestamp(fp, TIME); //set time stamp
	INC_TIME; //increase time
	for(int i = 0; i < n; i++) {
		volume[curb+i] = input[i];
	}
	return WRITE_SUCCESS;
}
__device__ uchar read(uchar *output, int n, uchar fp) {
	if(fp == ERROR) return READ_ERROR;
	if(n > BLOCK_SIZE) return READ_ERROR;
	int curb = BLOCK_LOC(fp);
	for(int i = 0; i < n; i++) {
		output[i] = volume[curb+i];
	}
	return READ_SUCCESS;
}
__device__ void gsys(uchar arg, const char *file) {
	if(arg == LS_S) {
		printf("===sorted by file size===\n");
		int a[MAX_CAPACITY];
		int n = read2bytes(curi+3);
		for(int i = 1; i < n; i++) {
			a[i-1] = INODE_LOC(curb + i*2);
		}
		n--;
		for(int i = 0; i < n; i++) {
			for(int j = i+1; j < n; j++) {
				if(read2bytes(a[i]+5) > read2bytes(a[j]+5)) {
					int tmp = a[i];
					a[i] = a[j];
					a[j] = tmp;
				}
			}
		}
		for(int i = 0; i < n; i++) {
			if(volume[a[i]+28]) printf("%s d\n", volume+a[i]+7);
			else printf("%s %d\n", volume+a[i]+7, read2bytes(a[i]+3));
		}
	}
	else if(arg == LS_D) {
	}
	else if(arg == PWD) {
		my_pwd(CUR_DIR);
	}
	else if(arg == CD) {
		int curi = INODE_LOC(CUR_DIR);
		int n = read2bytes(curi+3);
		int curb = BLOCK_LOC(CUR_DIR);
		for(int i = 1; i < n; i++) {
			int j = read2bytes(curb + i*2);
			int curi = INODE_LOC(j);
			if(my_strcmp(volume+curi+7, file) == 0) {
				if(volume[curi+28]) { //is a dir
					write2bytes(j, CUR_DIR_LOC);
				}
				else printf("%s not a directory!\n", file);
				return;
			}
		}
		printf("No such directory: %s\n", file);
	}
	else if(arg == CD_P) {
		if(CUR_DIR == 0) printf("Already at root\n");
		else {
			int cur = BLOCK_LOC(CUR_DIR);
			int father_i = read2bytes(cur);
			write2bytes(father_i, CUR_DIR_LOC);
		}
	}
	else if(arg == RM) { //NOT EVEN TOUCHED!!
		int i;
		for(i = 0; i < INODE_COUNT; i++) {
			int cur = INODE_LOC(i);
			if(isempty(i)) { //not empty
				if(my_strcmp(volume+cur+7, file) == 0) {
					if(volume[cur+28]) { //isdir
						printf("Can't remove a dir by rm.\n");
						return;
					}
					else break;
				}
			}
		}
		if(i == INODE_COUNT) printf("No such file %s!\n", file);
		else {
			free_room(i);
		}
	}
}
