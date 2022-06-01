---
layout: post
title:  "Welcome to Jekyll!"
date:   2022-04-28 22:02:57 -0400
categories: jekyll update
---

之前有打算用开一个blog的想法，然后我其实一点也不会web-programming，虽然面向搜索引擎编程能整一些非常简陋的网页，但要做博客的话，还是要求大概是能比较简单的编辑（最好自带文本编辑器），有模板，同时因为是个人博客最好界面能高度自定义，一开始想的是[wordpress.com][wordpress.com], 但这太不geeeeeeek了（因为虽然wordpress是一个开源软件，但是wordpress.com是一个偏向商业的面向个人的博客托管服务，是会有广告位置的），我还是想利用github page去搭， 而且wordpress.com上如果要使用插件是要收费的。

一开始想得还是非常美好的：那我就在本地搭一个wordpress（但wordrpess是动态网页，是有数据库的）然后想办法把他静态化就好了，毕竟所有的文件都在本地，wordpress（不管是wordpress.com还是wordpress）也确实提供导出服务。 但后来发现网页上的内容和素材其实是被存在数据库里面了，并且官方的导出的文件格式主要是供在不同服务器上使用wordpress去做迁移的时候用的（或者从wordpress.com迁移到自己的服务器），对静态化不是很友好。 

wordpress是有一些静态化插件的，但对于在docker容器里面运行的并不是很友好，我最后是用了[winhtttrack][https://www.httrack.com/], 也就是每次在wordpress上写，然后用这个软件静态化，最后再git push


[jekyll-docs]: https://jekyllrb.com/docs/home
