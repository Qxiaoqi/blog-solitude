---
title: 浏览器重绘和requestAnimationFrame
date: 2020-05-07 16:18:19
toc: true
categories:
- 前端
tags:
- 浏览器
cover: https://file-1305436646.file.myqcloud.com/blog/banner/gui-dang.webp

---

又是很久没写文章了，简单总结一下从去年八月到现在将近一年的事吧，去年八月那会儿当时在学脚手架相关的部分，当时实习了一个月就离职准备秋招了，然后在秋招也拿到了满意的offer了。然后就是大四寒假了，写了一个外卖平台的小程序，差不多写了快两个月吧，中间也确实遇到一些比较有意思的问题，后面会简单整理一下遇到的问题。然后就是毕业设计和毕业论文了。差不多项目上主要做了这些事情吧，小程序和毕业设计也算是都有所收获吧。

然后就是回到本文中的标题了，因为被分到了小游戏的组，入职后可能会做一些工具上的开发。因为这方面实在是了解不多，也是趁着论文写完赶紧先学习一下相关的部分吧。

<!--more-->

## 思考

其实目前根据个人粗浅的理解，小游戏就是不同帧数绘制不同的物体，中间可能会有一些用户的交互来改变这些物体的位置等等。

那么虽然微信小游戏和浏览器的环境不同，但是也有相似之处，因为了解不多，这里暂且先归为一类。那么图像的绘制必然要涉及到浏览器的绘制部分。

其实经历了春招秋招，大家对浏览器的工作原理比如语法分析、构建DOM、重绘重排等等流程肯定都比较了解了。这里就不再多说，但是目前又产生了一些疑问。首先`requestAnimationFrame`的使用就是在浏览器绘制之前会产生回调，因为浏览器的一些策略等等，大概是 1000ms / 60 近似等于 16.7ms 的频率绘制。

* 但是这里每经过 16.7ms 浏览器一定会绘制吗？这里的浏览器绘制和重绘是一个概念吗？

* 如何捕捉浏览器何时进行绘制 或者 何时进行重绘？使用 requestAnimationFrame来捕捉？

首先，必须要知道，浏览器有自己的一些优化策略，比如连续的更改DOM的颜色，按照对个人对重绘的理解，更改颜色是会触发重绘的，更改DOM位置等会触发回流，回流必然重绘。但是假如连续的更改两次颜色（css要指定`transition`动画），比如

```js
document.getElementById('test').style.backgroundColor = 'red'
document.getElementById('test').style.backgroundColor = 'blue'
```

诸如上面代码其实并不会看到颜色的变化，即只能看到蓝色，而不能看到红色，那么明显浏览器对这部分在一定时间会收集所有操作，然后一次改变，我暂时认为这个时间是浏览器绘制的频率时间即16.7ms为界。

然后根据对回流重绘的了解，诸如获取布局尺寸等信息的时候，为了保证准确性会强制触发回流重绘，比如`offsetTop`、`getComputedStyle()`等等[相关资料]('https://juejin.im/post/5e8ec67ce51d4546fd4813d3')。

因此我们做如下改动

```js
document.getElementById('test').style.backgroundColor = 'red'
console.log("offsetTop:", document.getElementById('test').offsetTop)
document.getElementById('test').style.backgroundColor = 'blue'
```

可以发现颜色发生了渐变，那么可以得出会触发浏览器重新绘制，但是问题来了，我如何去得出浏览器在执行`offsetTop`的时候发生了重新绘制？根据我目前所掌握的，我发现似乎只有`requestAnimationFrame`有可能能捕捉这个过程，按照我原本的理解，由于触发重新绘制，所以这个时候会触发`requestAnimationFrame`的回调，因此我写下了下面的代码

```js
    let number = 1
    let date = new Date().getTime()
    window.requestAnimationFrame(count)
    document.getElementById('test').style.backgroundColor = 'red';
    console.log("offsetTop:", document.getElementById('test').offsetTop)

    // for (let i = 0; i < 10000; i++) {
    //   console.log("offsetTop:", document.getElementById('test').offsetTop)
    // }

    document.getElementById('test').style.backgroundColor = 'blue';

    function count() {
      console.log("offsetTop:", document.getElementById('test').offsetTop)
      let now = new Date().getTime()
      let res = now - date
      date = now
      console.log("res:", res)
      number++
      
      window.requestAnimationFrame(count)
    }
```

按照我原先的想法，在count函数中立即触发`offsetTop`来立即刷新浏览器绘制，此时输出的res数值一定会有介于0-16.7ms之间的。但是在写下这篇文章的过程中，我又突然想到，16.7ms是浏览器基于硬件限制做出的界定，因此可以说16.7ms是最理想的刷新情况，因此应该是不可能出现小于16.7ms的情况。

那么问题来了，`offsetTop`触发回流重绘如何体现？

将中间注释掉的代码加上后，会发现第一次`requestAnimationFrame`回调大概发生在600ms左右，因此说明浏览器绘制延迟了很久。

在吃饭的时候，我发现我其实想清楚了一个问题，那就是浏览器重绘(repaint)和浏览器的绘制(paint)似乎是两个概念。借用一下[浏览器渲染原理文中的分析]('https://juejin.im/post/5d707d48f265da03c23ef4aa')

<img src="https://file-1305436646.file.myqcloud.com/blog/2020-5-7/1.jpg">

<img src="https://file-1305436646.file.myqcloud.com/blog/2020-5-7/2.jpg">

<img src="https://file-1305436646.file.myqcloud.com/blog/2020-5-7/3.jpg">

可以看出按照文中的分析，重绘是一个整体的过程，而按照个人的理解requestAnimationFrame所指的浏览器刷新频率是一个固定频率刷新的东西，这里我认为由于个人知识的局限，暂时没有必要将其的概念分析的很清晰，目前需要清楚的地方：

* 浏览器重绘是一个过程，涉及到计算样式、排版计算、绘制、渲染层合并等流程的整体过程

* requestAnimationFrame是一个浏览器根据硬件限制所做出的刷新频率回调（虽然似乎并不一定会固定回调，比如上文中代码的测试）

暂时清楚这两个点目前来讲已经可以，具体的区别与联系目前看别人的技术文章似乎已经没有办法回答我的个人疑惑，只等后面这方面技术积累更多的时候，有更加多的想法的时候再来看这个问题，也许能有不同的见解。
