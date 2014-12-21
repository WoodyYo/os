#Compiling Issue
直接 $ make main即可

執行程式則 $./ffs

#MEMO
1 block = 1 KB, totally 1024 blocks

##Super
留兩個byte來紀錄第一個空的inode

##Inode
直接對應block，爽！

	--------------------------------------------------------------------------------
	| 1 byte | 2 bytes        | 2 bytes        | 2 bytes      | 21 bytes | 1 byte  |
	|        |                |                |              |          |         |
	| empty? | fp/next empty  | size/capacity  | timestamp    | name     | is dir? |
	--------------------------------------------------------------------------------
	           (fp is never
			    used ="=)

INODE\_SIZE = 29

##Directory
一個block有1024 bytes，可以裝64個inode位址（共1024個inode，可用16 bits來表示，1024/16=64）
Meet the spec!!

	-----------------
	|pt to 1st empty|
	|---------------|            ------------------------------------------------------------
	| ..            | 2 bytes => |     6 bits point to next empty      | 10 bits point inode|
	|---------------|            ------------------------------------------------------------
	|a.txt          | 2 bytes           (if not empty, 6 bits = 0)
	|---------------|
	|       .       |   .
	|       .       |   .
	|       .       |   .
	-----------------

實驗證實pwd的遞迴可以跑大約18層，遠大於spec惹~~
