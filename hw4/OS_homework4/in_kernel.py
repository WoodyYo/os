def test():
	print "u32 fp;"
	for i in range(0, 1024):
		print 'fp = open("'+str(i)+'.txt", G_WRITE);'

	print "gsys(LS_D);"
	
	for i in range(0, 1024):
		if i%2 == 0:
			print 'gsys(RM, "' + str(i) + '.txt");'
	
	print "gsys(LS_D);"

