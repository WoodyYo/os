#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 這台該死的機器竟然沒有裝ruby?????????
from in_kernel import test

fp = open("main.cu")
while 1:
	s = fp.readline()
	if s.endswith("\n"):
		s = s[0:-1]
	print s
	if s.find("//####kernel start####") != -1:
		break
#===================================================#
test()
#===================================================#
while 1:
	s = fp.readline()
	if s.find("//####kernel end####") != -1:
		print s
		break
	elif s == '':
		print "WRONG FORMAT!!"
		break

while 1:
	s = fp.readline()
	if s == '':
		break
	if s.endswith("\n"):
		s = s[0:-1]
	print s
