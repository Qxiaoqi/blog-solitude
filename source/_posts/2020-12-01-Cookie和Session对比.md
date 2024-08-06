---
title: Cookie 和 Session 对比
date: 2020-12-01 19:18:19
toc: true
categories:
  - 面试整理
tags:
  - 浏览器
cover: https://file-1305436646.file.myqcloud.com/blog/banner/gui-dang.webp
---

# Cookie 和 Session 对比

## Cookie 和 Session

- Cookie： Cookie 是客户端保存用户信息的一种机制，用来记录用户的一些信息。存放在客户端，上限（4KB）。
- Session（有状态）：典型场景购物车，服务端需要为用户创建特定的 Session，用于标识用户。可以存在内存、数据库里面。

## Cookie

服务器通过设置 set-cookie 这个响应头，将 cookie 信息返回给浏览器，浏览器将响应头中的 cookie 信息保存在本地，当下次向服务器发送 HTTP 请求时，浏览器会自动将保存的这些 cookie 信息添加到请求头中。

### Cookie 属性

- max-age
  - 过期时间有多长
  - 默认在浏览器关闭时失效
- expires
  - 到哪个时间点过期
- secure
  - 表示这个 cookie 只会在 https 的时候才会发送
- HttpOnly
  - 设置后无法通过在 js 中使用 document.cookie 访问
  - 保障安全，防止攻击者盗用用户 cookie
- domain
  - 表示该 cookie 对于哪个域是有效的。

### 如何在窗口关闭时删除 Cookie

cookie 默认 max-age:session 只能在浏览器关闭时删除 cookie，如果要在窗口关闭时删除 cookie，可以监听窗口关闭 onbeforeunload，然后手动设置过期时间删除 cookie

## Session

服务端标识用户需要用到 Cookie，在 Cookie 中记录 Session ID。如果浏览器禁用，可以 URL 重写，即每次 HTTP 交互都在 URL 后面加上一个例如 sid=xxxxx 的参数

### 分布式 Session 一致性如何解决？

场景：多台服务器情况下，用户请求服务器 A，拿到 Session 后。再次请求服务器 B，服务器 B 中没有该用户的 Session，因此需要再次创建新的 SessionID，这就是产生的问题。

- cookie 来存 session（安全性并不可靠，而且 cookie 有大小限制 4KB，无法存更多东西）
- 使用单台服务器保存所有 Session，但是缺点是依赖程度比较高
- Nginx 配 ip_hash（得到 ip 后通过 hash 函数得到一个数值，然后对服务器列表大小进行取模运算，结果就是服务器序号），同一个 ip 只能在指定的机器上访问，可以解决 Session。（Upstream 模块实现负载均衡）
- 存到数据库里，同步 Session，效率不太高
- 存放在 redis 里
- token 代替（JWT），因为 JWT 方案不需要服务端存 Session，自然也就没这个问题了。

### Session 加密/解密过程

#### 若本次 cookie 中没有 connect.sid，则生成一个 [用 secret 生成 connect.sid]

1. 用 uid-safe 生成一个唯一 id，记为 sessionid，保证每次不重复；
2. 把上面的 connect.sid 制作成 's:' + sessionid + '.' + sessionid.sha256(secret).base64() 的形式，实现在 node-cookie-signature 的 sign 函数；
3. 把 sessionid 用 set-cookie 返回给前端；

#### 若本次 cookie 中包含 connect.sid，则验证它是否是本服务器生成的 [用 secret 验证 connect.sid]

1. 取出 cookie 中的 connect.sid，形式是上面的 's:' + sessionid + '.' + sessionid.sha256(secret).base64() ；
2. 从 connect.sid 中截取出 sessionid=connect.sid.slice(2, connect.sid.indexOf(’.’))；
3. 用取出的 sessionid 再算一次 sessionid.sha256(secret).base64() 记为 mac；
4. 截取 connect.sid 中’.'后的部分与 mac 对比；node-cookie-signature 的 unsign 函数（用上次计算的 sha256 值和这次计算的 sha256 值进行比较，只要 secret 一样，结果就一样）；
5. 验证成功的 sessionid 继续往下走。
