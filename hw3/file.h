#include <stdio.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

int load_binaryFile(const char *filename, uchar *a, int max_size);
void write_binaryFIle(const char *filename, uchar *a, int size);
