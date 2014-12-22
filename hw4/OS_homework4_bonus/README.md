#Compiling Issue
直接 $ make即可

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


##Something else
為什麼不寫個限制存取的API當作volume，要用甚麼cuda?請問這是cuda課嗎?用本來的C語言來寫互動性不是會更好嗎?一些可以在memory裡面用standard library或recursive處理的東西，現在全部要在cuda裡面跑，完全問號。每次編譯都一堆stack overflow的警告，煩死了。
