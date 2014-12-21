default:
	nvcc -arch=sm_30 test.cu -o ffs
main:
	nvcc -arch=sm_30 main.cu -o ffs
clean:
	rm -f ffs snapshot.bin test*
