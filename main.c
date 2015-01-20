#include <linux/fs.h>
#include <linux/init.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <linux/ioport.h>
#include <linux/errno.h>
#include <linux/mm.h>
#include <linux/unistd.h>
#include <asm/uaccess.h>
#include <asm/io.h>

#include "ioc_hw5.h"

#define DEV_NAME "mydev"
#define BUFFER_SIZE 15

#define DMA_BUFSIZE 64 
#define DMASTUIDADDR 0x0		//student ID
#define DMARWOKADDR  0x4		//RW function complete
#define DMAIOCOKADDR 0x8		//ioctl function complete
#define DMAIRQOKADDR 0xc		//ISR function complete
#define DMACOUNTADDR 0x10		//interrupt count function complete
#define DMAANSADDR	 0x14		//Computation ans
#define DMAREADABLEADDR 0x18	//READABLE variable for synchronization
#define DMABLOCKADDR 0x1c		//Blocking or Non-Blocking IO
#define DMAOPCODEADDR 0x20		//data.a opcode
#define DMAOPERANDBADDR 0x21	//data.b operand1
#define DMAOPERANDCADDR 0x25	//data.c operand2

#define MSG(format, arg...) printk(KERN_INFO "OS_HW5:%s() " format "\n", __FUNCTION__, ## arg)

MODULE_LICENSE("GPL");

struct cdev *dev_cdevp = NULL;
int dev_major, dev_minor;
static struct workqueue_struct *work_queue;
void* DMA_buffer;

struct data_in {
	char a;
	int b;
	short c;
};

int drv_open(struct inode *inode, struct file *filp);
int drv_release(struct inode *inode, struct file *filp);
ssize_t drv_write(struct file *filp, const char *buff, size_t count, loff_t *f_pos);
ssize_t drv_read(struct file *filp, char *buff, size_t count, loff_t *f_pos);
long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);
static void arithmetic_routine(struct work_struct *ws);
unsigned int myinl(unsigned short int offset);
void myoutl(unsigned int data, unsigned short int offset);
unsigned char myinb(unsigned short int offset);
void myoutb(unsigned int data, unsigned short int offset);
unsigned short myinw(unsigned short int offset);
void myoutw(unsigned int data, unsigned short int offset);

struct file_operations drv_fops = {
	.owner = THIS_MODULE,
	.read = drv_read,
	.write = drv_write,
	.unlocked_ioctl = drv_ioctl,
	.open = drv_open,
	.release = drv_release
};

int init_modules(void) {
	dev_t dev;
	int ret;
	MSG("..............Start..................");
	ret = alloc_chrdev_region(&dev, 0, 1, DEV_NAME);
	if(ret) {
		MSG("register faied");
		return -1;
	}
	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);
	MSG("register chardev(%d, %d)", dev_major, dev_minor);
	dev_cdevp = kzalloc(sizeof(struct cdev), GFP_KERNEL);
	if(dev_cdevp == NULL) {
		MSG("kzalloc dev faied");
		return -1;
	}
	cdev_init(dev_cdevp, &drv_fops);
	dev_cdevp->owner = THIS_MODULE;
	ret = cdev_add(dev_cdevp, MKDEV(dev_major, dev_minor), 1);
	if(ret < 0) {
		kfree(dev_cdevp);
		MSG("add chr dev failed");
		return -1;
	}
    DMA_buffer = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
	if(!DMA_buffer) {
		MSG("kzalloc dma failed");
	}
	else MSG("allocate dma buffer");
    work_queue = create_workqueue("wq");

	return 0;
}

void exit_modules(void) {
	dev_t dev;
	dev = MKDEV(dev_major, dev_minor);
	if(dev_cdevp) {
		cdev_del(dev_cdevp);
		kfree(dev_cdevp);
	}
	unregister_chrdev_region(dev, 1);
	MSG("unregister chrdev");
	MSG("..............End..................");
}

module_init(init_modules);
module_exit(exit_modules);

int drv_release(struct inode *inode, struct file *filp) {
	MSG("device close");
	return 0;
}
int drv_open(struct inode *inode, struct file *filp) {
	MSG("device open");
	return 0;
}
ssize_t drv_write(struct file *filp, const char *buff, size_t count, loff_t *f_pos) {
	struct data_in *data;
	static struct work_struct *work;
	data = (struct data_in*) kzalloc(sizeof(char)*DMA_BUFSIZE, GFP_KERNEL);
	if(copy_from_user(data, buff, count) < 0) {
		kfree(data);
		return -1;
	}
	myoutb(data->a, DMAOPCODEADDR);
	myoutl(data->b, DMAOPERANDBADDR);
	myoutw(data->c, DMAOPERANDCADDR);
	work = (struct work_struct*) kzalloc(sizeof(work), GFP_KERNEL);
	INIT_WORK(work, arithmetic_routine);
	queue_work(work_queue, work);
	MSG("queue work");
	if(myinl(DMABLOCKADDR)) {
		MSG("block");
		if(count > BUFFER_SIZE) count = BUFFER_SIZE;
		while(!myinl(DMAREADABLEADDR)) msleep(100);
	}
	kfree(data);
	kfree(work);
}
ssize_t drv_read(struct file *filp, char *buff, size_t count, loff_t *f_pos) {
	int ans = myinl(DMAANSADDR);
	copy_to_user(buff, &ans, count);
	MSG("ans = %d", ans);
	return count;
}
long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	int data;

	if(_IOC_TYPE(cmd) != HW5_IOC_MAGIC) return -1;
	if(_IOC_NR(cmd) > HW5_IOC_MAXNR) return -1;

	data = *((int*) arg);

	if(cmd == HW5_IOCSETSTUID) {
		myoutl(data, DMASTUIDADDR);
		MSG("My STUID is = %d", data);
	}
	else if(cmd == HW5_IOCSETRWOK) {
		myoutl(data, DMARWOKADDR);
		MSG("RW OK");
	}
	else if(cmd == HW5_IOCSETIOCOK) {
		myoutl(data, DMAIOCOKADDR);
		MSG("IOC OK");
	}
	else if(cmd == HW5_IOCSETIRQOK) {
		myoutl(data, DMAIRQOKADDR);
		MSG("IRQ OK");
	}
	else if(cmd == HW5_IOCSETBLOCK) {
		myoutl(data, DMABLOCKADDR);
		if(data) MSG("Blocking IO"); 
		else MSG("Non-Blocking IO");
	}
	else if(cmd == HW5_IOCWAITREADABLE) {
		while(!myinl(DMAREADABLEADDR)) {
			msleep(100);
		}
		MSG("wait readable %d", myinl(DMAREADABLEADDR));
		data = myinl(DMAREADABLEADDR);
		copy_to_user(&arg, &data, sizeof(int));
	}
	return 0;
}

static void arithmetic_routine(struct work_struct *ws) {
	int fnd = 0;
	int i, num, isPrime, base, nth;
	myoutl(0, DMAREADABLEADDR);
	base = myinl(DMAOPERANDBADDR);
	nth = myinw(DMAOPERANDCADDR);
	num = base;
	while(fnd != nth) {
		isPrime = 1;
		num++;
		for(i = 2; i <= num/2; i++) {
			if(num % i == 0) {
				isPrime = 0;
				break;
			}
		}
		if(isPrime) {
			fnd++;
		}
	}
	MSG("%d p %d = %d", base, nth, num);
	myoutl(num, DMAANSADDR);
	myoutl(1, DMAREADABLEADDR);
}

unsigned int myinl(unsigned short int offset) {
	return *(unsigned int*)(DMA_buffer+offset);
}
void myoutl(unsigned int data, unsigned short int offset) {
	*(unsigned int*)(DMA_buffer+offset) = data;
}
unsigned char myinb(unsigned short int offset) {
	return *(unsigned char*)(DMA_buffer+offset);
}
void myoutb(unsigned int data, unsigned short int offset) {
	*(unsigned char*)(DMA_buffer+offset) = data;
}
unsigned short myinw(unsigned short int offset) {
	return *(unsigned short*)(DMA_buffer+offset);
}
void myoutw(unsigned int data, unsigned short int offset) {
	*(unsigned short*)(DMA_buffer+offset) = data;
}
