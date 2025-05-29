---
layout: "post"
title:  "使用TailScale进行内网穿透，PC as server"
date:   2025-05-29 12:08:00 -0500
categories: jekyll update
---

本片的内容受到[五道口纳什：GPU服务器](https://www.bilibili.com/video/BV1PYfpYdEPx/?spm_id_from=333.1387.search.video_card.click)启发，并且手边因为课程需要刚好有Raspberry Pi，适合将PC改造成个人用的GPU服务器，或者个人用NAS。实操中发现，Tailscale已经提供了比较好的内网穿透解决方案，并且不需要额外购置域名。

#### Tailscale + SSH server

在Windows 11上，可以在设置中安装SSH server（`System >> Optional Features >> OpenSSH Server`）。通过局域网链接后可以成为正常使用PC上的命令行。ssh服务器的密码是微软账号的密码（而非pc的登录密钥），因此，如果通过谷歌邮箱登录的微软账号需要另外设置一个账户密码以便使用。

为了跨局域网进行链接，我们使用tailscale进行内网穿透之后，可以通过tailscale提供的ip链接上对应的机器（或和在同一个局域网中一样，直接使用device name）：

~~~
C:\Users\Yitao Yu>tailscale status
100.93.159.83   laptop-4fqohibt      yitao-yu@    windows -
100.99.135.112  yitaopc              yitao-yu@    windows idle
100.119.212.60  yitaoraspberrypi     yitao-yu@    linux   idle
~~~

对于个人用户而言，这是免费的。

#### RaspberryPi通过Wake On LAN唤醒PC

RaspberryPi位于和PC在同一局域网，并且也可以从笔记本通过tailscale链接上, 我们可以通过raspberry pi向pc的以太网适配器发送信号来打开pc。这样的话，PC可以长期处于关闭状态，而Raspberry Pi的功率较小，耗电可以忽略不计。

需要注意的是，WakeOnLAN需要在PC的BIOS中找到相关选项进行设置。

在Pi上使用`sudo apt install wakeonlan`并使用`wakeonlan [MAC]`唤醒PC。MAC地址可以在PC上使用`ipconfig \all`并且找到"Ethernet adapter Ethernet"下的"Physical Address"找到。（注意将"-"变为":"链接）这一步，可以在笔记本上使用`wakemeonlan`软件进行debug。

#### VNC

考虑到未来通过GUI安装特定软件的需要，使用TightVNC或其他VNC类的软件进行串流。在tightvnc客户端上，使用device name加端口号即可（而非tailscale的ip地址），在这一步发现设置密码后在tight vnc客户端上会报错，所以设置为免密登录。

如果要实现局域网内的远程游戏，可以使用sunshine/moonlight等支持更高帧率和分辨率的软件进行串流，但是内网穿透的环境下需要考虑到tailscale的带宽负担（请适度薅羊毛）。