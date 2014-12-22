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
#define MKDIR 6
#define RM_RF 7

#define TIME_LOC DATA_START-2
#define TIME (read2bytes(TIME_LOC))
#define INC_TIME (write2bytes(TIME+1, TIME_LOC))

#define CUR_DIR_LOC DATA_START-4
#define CUR_DIR (read2bytes(CUR_DIR_LOC))
#define SET_CUR_DIR(i) (write2bytes(i, CUR_DIR_LOC))

#define MAX_CAPACITY 60

typedef unsigned char uchar;
typedef uint32_t u32;

__device__ __managed__ uchar *volume;

int load_binaryFile(const char *filename, uchar *a, int max_size);
void write_binaryFIle(const char *filename, uchar *a, int size);
__device__ u32 open(char *name, uchar mode);
__device__ uchar write(uchar *input, int n, u32 fp);
__device__ uchar read(uchar *output, int n, u32 fp);
__device__ void gsys(uchar arg, char* file=NULL);
__device__ void init_volume();

__device__ int my_strcmp(uchar *a, char *b) {
	for(int i = 0; i < NAME_LENGTH; i++) {
		if(a[i] == 0 && b[i] == 0) return 0;
		else if(a[i] != b[i]) return 1;
	}
	return 1;
}
__device__ void my_strcpy(uchar *d, char *s) {
	int i;
	int memo = 0;
	for(i = 0; i < NAME_LENGTH-1; i++) {
		if(s[i] == 0) break;
		else if(s[i] == '/') memo = i+1;
	}
	s = s+memo;
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
__device__ uchar* name(int i, char *v=NULL) {
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
__device__ int inode_id(int i, int num, int v=-1) {
	//直接call這個函數，就有next_empty_child(i, num, 0)的效果
	if(num == ERROR) return ERROR;
	int cur = BLOCK_LOC(i);
	if(v == -1) return read2bytes(cur + num*2);
	else write2bytes(v, cur + num*2);
	return 0;
}
__device__ int next_empty_child(int i, int num, int v=-1) {
	int cur = BLOCK_LOC(i);
	if(v == -1) return (volume[cur + num*2]>>2);
	else volume[cur + num*2] = (v<<2);
	return 99;
}
__device__ int find_room(int dir) {
	int i = read2bytes(0);
	int j = next_empty(i);
	write2bytes(j, 0);
	//put the file in dir
	int ii = next_empty_child(dir, 0);
	int jj = next_empty_child(dir, ii);
	next_empty_child(dir, 0, jj);
	inode_id(dir, ii, i); //point ii in dir to "inode i"

	timestamp(dir, TIME);
	INC_TIME;
	return i;
}
__device__ void free_room(int tar, int dir) { //tar is "在 dir 中的位置", not abs id!!!
	//free from directory
	int i = next_empty_child(dir, 0);
	next_empty_child(dir, 0, tar);
	next_empty_child(dir, tar, i);
	//free inode
	int id = inode_id(dir, tar);
	i = read2bytes(0);
	next_empty(0, id);
	next_empty(id, i);

	timestamp(dir, TIME);
	INC_TIME;
}
__device__ int find_by_name(char *file, int dir, int *last_dir) { //return "在dir中的位置"!!!
	*last_dir = dir; //兒子會把媽媽蓋掉!
	char ch;
	char s[NAME_LENGTH]; //String processing in C is garbage!
	for(int n = 0; ;n++) {
		ch = file[n];
		if(ch == 0) {
			s[n] = 0;
			for(int i = 2; i < MAX_CAPACITY; i++) {
				int id = inode_id(dir, i);
				if(next_empty_child(dir, i) == 0 && my_strcmp(name(id), s) == 0) {
					return i; //no matter dir or file.
				}
			}
			return ERROR;
		}
		if(ch == '/') { //往下層
			s[n] = 0;
			for(int i = 2; i < MAX_CAPACITY; i++) {
				int id = inode_id(dir, i);
				if(next_empty_child(dir, i) == 0 && my_strcmp(name(id), s) == 0) {
					if(isdir(id)) return find_by_name(file+n+1, id, last_dir);
					else break;
				}
			}
			*last_dir = ERROR;
			return ERROR;
		}
		else s[n] = ch;
	}
}
__device__ void free_dir(int tar_dir, int dir) {
	for(int i = 2; i < MAX_CAPACITY; i++) {
		if(next_empty_child(tar_dir, i) == 0) {
			int id = inode_id(tar_dir, i);
			if(isdir(id)) free_dir(id, tar_dir);
			else free_room(i, tar_dir);
		}
	}
	free_room(tar_dir, dir);
}
__device__ void my_pwd(int i) {
	if(i == 0) {
		return;
	}
	int father = inode_id(i, 1);
	my_pwd(father);
	printf("/%s", name(i));
}

__global__ void mykernel(uchar *input, uchar *output) {
	init_volume();
	//####kernel start####
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
	write2bytes(1, 0); //first empty = 1
	// create root dir
	isempty(0, 1); //not empty
	size(0, 0); //capacity
	timestamp(0, 0); //timestamp
	name(0, "/\0");
	isdir(0, 1); //is a dir

	next_empty_child(0, 0, 2);  //0 is superblock, 1 is ./ , starts from 2
	for(int i = 2; i < MAX_CAPACITY; i++) {
		next_empty_child(0, i, i+1); //point to next
	}
	SET_CUR_DIR(0); //set cur_dir to root

	INC_TIME; //time init

	for(int i = 1; i < INODE_COUNT; i++) {
		isempty(0, 0); //set empty
		next_empty(i, i+1); //point to i+1
	}
}
__device__ u32 open(char *file, uchar mode) {
	int last_dir;
	int i = find_by_name(file, CUR_DIR, &last_dir);
	int id = inode_id(last_dir, i);
	if(last_dir == ERROR) {
		printf("Wrong path %s\n", file); //連路徑都打錯太扯了，就算是G_WRITE都不能給過
		return ERROR;
	}
	else if(id != ERROR) return id;
	else if (mode == G_WRITE) { //create
		int id = find_room(last_dir);
		if(id == -1) return ERROR;
		isdir(id, 0); //set not directory
		isempty(id, 1); //set not empty
		next_empty(id, 0); //set fp to 0
		size(id, 0); //set size to 0
		timestamp(id, TIME); //set timestamp
		name(id, file); //set name
		INC_TIME; //increase time
		return id;
	}
	else return ERROR;
}

__device__ uchar write(uchar *input, int n, u32 fp) {
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
__device__ uchar read(uchar *output, int n, u32 fp) {
	if(fp == ERROR) return READ_ERROR;
	if(n > BLOCK_SIZE) return READ_ERROR;
	int curb = BLOCK_LOC(fp);
	for(int i = 0; i < n; i++) {
		output[i] = volume[curb+i];
	}
	return READ_SUCCESS;
}
__device__ void gsys(uchar arg, char *file) {
	if(arg == LS_S) {
		printf("===sorted by file size===\n");
		int a[MAX_CAPACITY];
		int n = 0;
		int cur_dir = CUR_DIR;
		for(int i = 2; i < MAX_CAPACITY; i++) { //從2開始，避開 ../
			if(next_empty_child(cur_dir, i) == 0)
				a[n++] = inode_id(cur_dir, i);
		}
		for(int i = 0; i < n; i++) {
			for(int j = i+1; j < n; j++) {
				if(isdir(a[j]) || size(a[i]) > size(a[j])) {
					int tmp = a[i];
					a[i] = a[j];
					a[j] = tmp;
				}
			}
		}
		for(int i = 0; i < n; i++) {
			if(isdir(a[i])) printf("%s d\n", name(a[i]));
			else printf("%s %d\n", name(a[i]), size(a[i]));
		}
	}
	else if(arg == LS_D) {
		printf("===sorted by modified time===\n");
		int a[MAX_CAPACITY];
		int n = 0;
		int cur_dir = CUR_DIR;
		for(int i = 2; i < MAX_CAPACITY; i++) {
			if(next_empty_child(cur_dir, i) == 0)
				a[n++] = inode_id(cur_dir, i);
		}
		for(int i = 0; i < n; i++) {
			for(int j = i+1; j < n; j++) {
				if(timestamp(a[i]) < timestamp(a[j])) {
					int tmp = a[i];
					a[i] = a[j];
					a[j] = tmp;
				}
			}
		}
		for(int i = 0; i < n; i++) {
			if(isdir(a[i])) printf("%s d\n", name(a[i]));
			else printf("%s\n", name(a[i]));
		}
	}
	else if(arg == PWD) {
		if(CUR_DIR == 0) printf("/");
		my_pwd(CUR_DIR);
		printf("\n");
	}
	else if(arg == CD) {
		int i, last_dir;
		if(file[0] == '/') i = find_by_name(file, 0, &last_dir); //abs
		else i = find_by_name(file, CUR_DIR, &last_dir);
		int id = inode_id(last_dir, i); //if i = ERROR, id = ERROR, too!
		if(id == ERROR) printf("No such directory %s\n", file);
		else if(!isdir(id)) printf("%s is s file\n", file);
		else SET_CUR_DIR(id);
	}
	else if(arg == CD_P) {
		int cur_dir = CUR_DIR;
		if(cur_dir == 0) printf("alread at root\n");
		else SET_CUR_DIR(inode_id(cur_dir, 1));
	}
	else if(arg == RM) {
		int last_dir;
		int i = find_by_name(file, CUR_DIR, &last_dir);
		int id = inode_id(last_dir, i);
		if(id == ERROR) printf("No such file %s\n", file);
		else if(isdir(id)) printf("%s is s directory, can't remove by RM\n", file);
		else free_room(i, last_dir); //free_room 吃的是i不是id!!
	}
	else if(arg == RM_RF) {
		int last_dir;
		int i = find_by_name(file, CUR_DIR, &last_dir);
		int id = inode_id(last_dir, i);
		if(id == ERROR) printf("No such file %s(though rm-rf is silence in real Linux XD.)\n", file);
		else if(!isdir(id)) printf("%s is a file, can't remove it by RM_RF(though we can in real Linux XD)\n", file);
		else free_dir(i, last_dir); //free_dir 吃的是i不是id!!
	}
	else if(arg == MKDIR) {
		int last_dir;
		int i = find_by_name(file, CUR_DIR, &last_dir);
		int id = inode_id(last_dir, i);
		if(last_dir == ERROR) printf("Wrong path %s\n", file);
		else if(id != ERROR) printf("%s exists!\n", file);
		else {
			id = find_room(last_dir);
			name(id, file);
			timestamp(id, TIME);
			INC_TIME;
			isdir(id, 1);
			isempty(id, 1);
			next_empty_child(id, 0, 2);  //0 is superblock, 1 is ../ , starts from 2
			inode_id(id, 1, last_dir);
			for(int i = 2; i < MAX_CAPACITY; i++) {
				next_empty_child(id, i, i+1); //point to next
			}
		}
	}
}
