$make 即可編譯，編譯後 $./vm即可。


* Reviews and Complaints

一個對作業系統完全沒基礎的死大學生 的抱怨，如果在高手眼中顯得很白癡還請見諒，我對memory management的認知僅止於這次的作業
先檢討自己
inverse pt裡面好像不需要存frame，可以直接用index判斷，不過ppt上開了u32這麼大的空間，不拿來存frame要幹嗎?
然後來檢討別的
最顯而易見的，為什麼data要開那麼大，去擠壓到pt的空間，搞到大家要用inverse pt?
讓人完全沒有用linked list去實作lru的動力，因為時間複雜度都被find吃掉了
OK，如果是助教用心良苦要讓我們學習inverse pt，才下這種自我閹割的限制，那我可以接受
第二，為什麼一開始要全部設為invalid?
如果在init裡面設 for(i = 0; i < n; i++) pt[i].p = i
然後全部設為valid，這樣第一次Gwrite就可以直接找到對應的page
如果找不到，再用lru也不遲，看不出init全設invalid的必要，甚至該說這樣反而拖累程式效能
最後，為什麼data是區域變數，pt是全域變數?
照理說，每個程式/線程有自己的pt，而共用一個physical memory不是比較合理嗎?
現在的實作方式，假如thread0用了100KB，其他thread都只用1KB，thread0的虛擬記憶體還是會爆，這樣合理嗎?
