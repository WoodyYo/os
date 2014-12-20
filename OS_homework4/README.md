#Compiling Issue
直接 $ make即可
執行程式則 $./fs

#MEMO
1 block = 1 KB, totally 1024 blocks

##Super
留兩個byte來紀錄第一個空的inode

##Inode
直接對應block，爽！

	---------------------------------------------------------------
	| 1 byte | 2 bytes        | 2 bytes | 2 bytes      | 21 bytes |
	|        |                |         |              |          |
	| empty? | fp/next empty  | size    | timestamp    | name     |
	---------------------------------------------------------------

INODE\_SIZE = 28

##Directory
一個block有1024 bytes，可以裝64個inode位址（共1024個inode，可用16 bits來表示，1024/16=64）
Meet the spec!!
