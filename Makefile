mname := mydev
$(mname)-objs := main.o
obj-m := $(mname).o

KERNELDIR := /lib/modules/`uname -r`/build

all:
	$(MAKE) -C $(KERNELDIR) M=`pwd` modules
	gcc -o test test.c

insert:
	$(MAKE) -C $(KERNELDIR) M=`pwd` modules
	sudo insmod mydev.ko
	gcc -o test test.c

clean:
	$(MAKE) -C $(KERNELDIR) M=`pwd` clean
	sudo rmmod mydev
	dmesg
	rm test
