#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 這台該死的機器竟然沒有裝ruby?????????
from in_kernel import test

fp = open("main.cu")
OUT = open("test.cu", 'w')
out = OUT

def output(s):
	OUT.write(s + "\n")

while 1:
	s = fp.readline()
	out.write(s)
	if s.find("//####kernel start####") != -1:
		break
#===================================================#
test(output)
#===================================================#
while 1:
	s = fp.readline()
	if s.find("//####kernel end####") != -1:
		out.write(s)
		break
	elif s == '':
		out.write("WRONG FORMAT!!")
		break

while 1:
	s = fp.readline()
	if s == '':
		break
	out.write(s)
