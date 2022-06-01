---
layout: postmarkdownp
title:  "Jekyll真是太香了!！"
date:   2022-05-31 22:05:55 -0400
categories: jekyll update
---

之前有打算用开一个blog的想法，然后我其实一点也不会web-programming，虽然面向搜索引擎编程能整一些非常简陋的网页，但要做博客的话，还是要求大概是能比较简单的编辑（最好自带文本编辑器），有模板，同时因为是个人博客最好界面能高度自定义，一开始想的是[wordpress.com][wordpress.com], 但这太不geeeeeeek了（因为虽然wordpress是一个开源软件，但是wordpress.com是一个偏向商业的面向个人的博客托管服务，是会有广告位置的），我还是想利用github page去搭， 而且wordpress.com上如果要使用插件是要收费的。

一开始想得还是非常美好的：那我就在本地搭一个wordpress（但wordrpess是动态网页，是有数据库的）然后想办法把他静态化就好了，毕竟所有的文件都在本地，wordpress（不管是wordpress.com还是wordpress）也确实提供导出服务。 但后来发现网页上的内容和素材其实是被存在数据库里面了，并且官方的导出的文件格式主要是供在不同服务器上使用wordpress去做迁移的时候用的（或者从wordpress.com迁移到自己的服务器），对静态化不是很友好。 

Wordpress是有一些静态化插件的，但对于在docker容器里面运行的并不是很友好()，我最后是用了[winhtttrack][https://www.httrack.com/], 也就是每次在wordpress上写，然后用这个软件静态化，最后再git push，这样一个workflow。 但太麻烦了。

然后大概又考虑了一下要不要跳出一下舒适区去看一下jekyll，结果jekyll实在是太香了，上手难度非常低。

[官方文档][jekyll-docs]

除了文档之外b站上有人搬运一个上手教程也挺好： [b站教程][b站教程]

社区里面有一些模板或者项目质量非常高，比如： 

[antarctica][antarctica] 非常华丽

[jekyll-admin][jekyll-admin] 如果在容器里面搭建jekyll服务的话，jekyll-admin似乎内置了编辑器。 虽然我目前是在用sublime text。

使用github repo的api或者第三方评论托管服务可以添加评论功能，我一开始还琢磨了一段时间，因为静态网站理论上是不支持评论的。这里有[教程][comment-tutorial]

暂时先这样，未来可能小改一下模板或者试着看一下能不能用github action去在这里的rss里面去更新一些其他网站的帖子。



[jekyll-docs]: https://jekyllrb.com/docs/home
[b站教程][https://www.bilibili.com/video/BV1qs41157ZZ?p=7]
[antarctica][https://sdtelectronics.github.io/jekyll-theme-antarctica/]
[jekyll-admin][https://jekyll.github.io/jekyll-admin/]
[comment-tutorial][https://medium.com/@raravi/adding-comments-to-a-static-site-31506e77fc41]