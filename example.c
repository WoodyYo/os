#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/delay.h>
#include <linux/fs.h>
//#include <linux/config.h>
#include <linux/ioport.h>
#include <linux/errno.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/cdev.h>
#include <linux/workqueue.h>
#include <asm/io.h>
#include <asm/uaccess.h>
#include <linux/unistd.h>

//ioctl commands definitions
#include "ioc_hw5.h"

//DMA buffer
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

MODULE_LICENSE("GPL");

struct dataIn
{
	char a;
	int b;
	short c;
};

//file operations
static void drv_read(struct file* fd, unsigned int* data, unsigned int size);
static void drv_write(struct file* fd, struct dataIn* data, unsigned int size);
static int drv_ioctl(struct file* fd, unsigned int cmd, unsigned int* arg);
int drv_open(struct inode *inode, struct file* fd);
int drv_release(struct inode *inode, struct file* fd);

static int  __init init_modules(void);
static void __exit exit_modules(void);

struct file_operations drv_fops=
{
	.owner  = THIS_MODULE,
	.read     = drv_read,
	.write    = drv_write,
	.unlocked_ioctl      = drv_ioctl,
	.open    = drv_open,
	.release= drv_release
};

struct cdev *dev_cdevp = NULL;
void* DMA_buffer;
int dev_major;
int dev_minor;
static int stop_wq;
static struct workqueue_struct *workqueue;

//work routine
static void arithmetic_routine(struct work_struct *ws);
int prime(int base, short nth);

//in and out function
void myoutb(unsigned char data, unsigned short int port);
void myoutw(unsigned short data, unsigned short int port);
void myoutl(unsigned int data, unsigned short int port);
unsigned char myinb(unsigned short int port);
unsigned short myinw(unsigned short int port);
unsigned int myinl(unsigned short int port);

//file operations definition
static void drv_read(struct file* fd, unsigned int* data, unsigned int size)
{
	int ans;
	ans = myinl(DMAANSADDR);
	copy_to_user(data, &ans,size);

	printk("OS_HW5:%s(): ans = %d\n", __FUNCTION__, ans);

	//set readable
	myoutl(0, DMAREADABLEADDR);
}

static void drv_write(struct file* fd, struct dataIn* data, unsigned int size)
{
    struct dataIn* tmp;
    tmp = kzalloc(sizeof(char)*DMA_BUFSIZE, GFP_KERNEL);
    static struct work_struct *arith_workqueue;

    copy_from_user(tmp, data, size);
	myoutb(tmp->a, DMAOPCODEADDR);
	myoutl(tmp->b, DMAOPERANDBADDR);
	myoutw(tmp->c, DMAOPERANDCADDR);

	if(myinl(DMABLOCKADDR))
	{
		//Blocking I/O
		arith_workqueue = kzalloc(sizeof(typeof(*arith_workqueue)), GFP_KERNEL);
		INIT_WORK(arith_workqueue, arithmetic_routine);
		//schedule_work(arith_workqueue);
		queue_work(workqueue, arith_workqueue);
		printk("OS_HW5:%s():queue work\n", __FUNCTION__);
		printk("OS_HW5:%s():block\n", __FUNCTION__);
		while(!myinl(DMAREADABLEADDR))
		{
            msleep(1000);
        }
		kfree(tmp);
		return;
	}
	else
	{
		//Non-blocking I/O
		arith_workqueue = kzalloc(sizeof(typeof(*arith_workqueue)), GFP_KERNEL);
		INIT_WORK(arith_workqueue, arithmetic_routine);
		//schedule_work(arith_workqueue);
		queue_work(workqueue, arith_workqueue);
		printk("OS_HW5:%s():queue work\n", __FUNCTION__);
		kfree(tmp);
		return;
	}
}

int prime(int base, short nth)
{
    int fnd=0;
    int i, num, isPrime;

    num = base;
    while(fnd != nth) {
        isPrime=1;
        num++;
        for(i=2;i<=num/2;i++) {
            if(num%i == 0) {
                isPrime=0;
                break;
            }
        }

        if(isPrime) {
            fnd++;
        }
    }
    return num;
}

static void arithmetic_routine(struct work_struct *ws)
{
    int ans;
    char cmd;
    int operand1;
    short operand2;
    cmd = myinb(DMAOPCODEADDR);
    operand1 = myinl(DMAOPERANDBADDR);
    operand2 = myinw(DMAOPERANDCADDR);

    switch(cmd) {
        case '+':
            ans=operand1+operand2;
            break;
        case '-':
            ans=operand1-operand2;
            break;
        case '*':
            ans=operand1*operand2;
            break;
        case '/':
            ans=operand1/operand2;
            break;
        case 'p':
            ans = prime(operand1, operand2);
            break;
        default:
            ans=0;
    }

    myoutl(ans,DMAANSADDR);
    printk("OS_HW5:%s(): %d %c %d = %d\n", __FUNCTION__,operand1,cmd,operand2,ans);
    stop_wq=0;
    //set readable
    myoutl(1,DMAREADABLEADDR);

}

static int drv_ioctl(struct file* fd, unsigned int cmd, unsigned int *arg)
{
	if (_IOC_TYPE(cmd) != HW5_IOC_MAGIC)
        return -ENOTTY;
    if (_IOC_NR(cmd) > HW5_IOC_MAXNR)
        return -ENOTTY;

    unsigned int data;
    copy_from_user(&data, arg, sizeof(int));

	switch(cmd)
	{
		case HW5_IOCSETSTUID:
			myoutl(data, DMASTUIDADDR);
			printk("OS_HW5:%s(): My STUID is = %d\n", __FUNCTION__, *arg);
			break;

		case HW5_IOCSETRWOK:
			myoutl(data, DMARWOKADDR);
			printk("OS_HW5:%s(): RW OK\n", __FUNCTION__);
			break;

		case HW5_IOCSETIOCOK:
			myoutl(data, DMAIOCOKADDR);
			printk("OS_HW5:%s(): IOC OK\n", __FUNCTION__);
			break;

		case HW5_IOCSETIRQOK:
			myoutl(data, DMAIRQOKADDR);
			printk("OS_HW5:%s(): IRQ OK\n", __FUNCTION__);
			break;

		case HW5_IOCSETBLOCK:
			myoutl(data, DMABLOCKADDR);
			if(data==1)
				printk("OS_HW5:%s(): Blocking IO\n", __FUNCTION__);
			else
				printk("OS_HW5:%s(): Non-Blocking IO\n", __FUNCTION__);
			break;

		case HW5_IOCWAITREADABLE:
			while(!myinl(DMAREADABLEADDR))
			{
                msleep(1000);
			}
			printk("OS_HW5:%s(): wait readable %d\n", __FUNCTION__, myinl(DMAREADABLEADDR));
			data = myinl(DMAREADABLEADDR);
			copy_o_user(arg, &data,sizeof(int));
			break;
	}
}

int drv_open(struct inode *inode, struct file* fd)
{
	printk("OS_HW5:%s(): device open\n", __FUNCTION__);
    return 0;
}

int drv_release(struct inode *inode, struct file* fd)
{
	printk("OS_HW5:%s(): device close\n", __FUNCTION__);
    return 0;
}

static int  __init init_modules(void)
{
	printk("OS_HW5:%s():...............start...............\n", __FUNCTION__);
	dev_t dev;
    int ret;

    ret = alloc_chrdev_region(&dev, 0, 1, "brook");
    if (ret) {
        printk("OS_HW5:%s():can't alloc chrdev\n",__FUNCTION__);
        return ret;
    }

    dev_major = MAJOR(dev);
    dev_minor = MINOR(dev);
    printk("OS_HW5:%s():register chrdev(%d,%d)\n", __FUNCTION__, dev_major, dev_minor);

    dev_cdevp = kzalloc(sizeof(struct cdev), GFP_KERNEL);
    DMA_buffer = kzalloc(DMA_BUFSIZE, GFP_KERNEL);

    //handle error
    if (dev_cdevp == NULL) {
        printk("OS_HW5:%s():kzalloc dev failed\n",__FUNCTION__);
        goto failed;
    }
    if(DMA_buffer == NULL)
    {
        printk("OS_HW5:%s():kzalloc dma buffer failed\n",__FUNCTION__);
        goto failed;
    }

	cdev_init(dev_cdevp, &drv_fops);
	printk("OS_HW5:%s():allocate dma buffer\n",__FUNCTION__);
    dev_cdevp->owner = THIS_MODULE;
    ret = cdev_add(dev_cdevp, MKDEV(dev_major, dev_minor), 1);
    if (ret < 0) {
        printk("OS_HW5:%s():add chr dev failed\n",__FUNCTION__);
        goto failed;
    }

    //work queue
    workqueue = create_workqueue("wq");

    return 0;

failed:
    if (dev_cdevp) {
        kfree(dev_cdevp);
        dev_cdevp = NULL;
    }
    return 0;
}

static void __exit exit_modules(void)
{
	dev_t dev;

    dev = MKDEV(dev_major, dev_minor);
    if (dev_cdevp) {
        cdev_del(dev_cdevp);
        kfree(dev_cdevp);
		printk("OS_HW5:%s():free dma buffer\n",__FUNCTION__);
    }
    unregister_chrdev_region(dev, 1);
    flush_workqueue(workqueue);
    stop_wq=1;
    destroy_workqueue(workqueue);
    printk("OS_HW5:%s():unregister chrdev\n",__FUNCTION__);
    printk("OS_HW5:%s():................end................\n", __FUNCTION__);
}

//in and out function definition
void myoutb(unsigned char data, unsigned short int port)
{
	//memcpy(DMA_buffer+port,&data,sizeof(int)/4);
    *(unsigned char*)(DMA_buffer+port) = data;
}

void myoutw(unsigned short data, unsigned short int port)
{
	//memcpy(DMA_buffer+port,&data,sizeof(int)/2);
	*(unsigned short*)(DMA_buffer+port) = data;
}

void myoutl(unsigned int data, unsigned short int port)
{
	//memcpy(DMA_buffer+port,&data,sizeof(int));
	*(unsigned int*)(DMA_buffer+port) = data;
}

unsigned char myinb(unsigned short int port)
{
	/*unsigned char tmp;
	memcpy(&tmp, DMA_buffer+port, sizeof(int)/4);
	return tmp;*/
	return (*(unsigned char*)(DMA_buffer+port));
}

unsigned short myinw(unsigned short int port)
{
	/*unsigned short tmp;
	memcpy(&tmp, DMA_buffer+port, sizeof(int)/2);
	return tmp;*/
	return (*(unsigned short*)(DMA_buffer+port));
}

unsigned int myinl(unsigned short int port)
{
	/*unsigned int tmp;
	memcpy(&tmp, DMA_buffer+port, sizeof(int));
	return tmp;*/
	return (*(unsigned int*)(DMA_buffer+port));
}

module_init(init_modules);
module_exit(exit_modules);
