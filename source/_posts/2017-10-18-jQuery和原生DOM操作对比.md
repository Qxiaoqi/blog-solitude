---
title: jQuery和原生DOM操作对比
date: 2017-10-18 20:43:17
toc: true
categories:
- 前端
tags: 
- JavaScript
- jQuery
cover: https://iph.href.lu/500x250
---

说到这个jQ库，我之前一直搞错一个概念，以为jQ是一个框架，但是查了一些资料后呢，发现它原来是个库。那么它们的区别是什么呢，本质区别在于控制反转。那么通俗点讲，库是一个封装好的集合，控制权在使用者。而框架是一个架构，向用户提供解决方案，控制权在框架。

那么不说那么专业的东西，说点我看完jQ的体会，发现jQ确实会让js编程变得简单很多，特别是Ajax的使用，jQ里面并不复杂，但是若要直接写原生js那是有点麻烦的。虽然这些东西很方便，但是我要说的是，一定不要依赖于库或者框架，前端技术发展很快，框架层出不穷，但是核心的js是不会变的，这些所有东西都是可以用原生js实现的。所以，学这些东西的时候最好再用js尝试实现相应的功能，这样才能游刃有余。

<!--more-->

## 一、jQ获取元素

jQ库与js原生代码的比较如下：
![](https://file-1305436646.file.myqcloud.com/blog/2017-10-18/hideButton.png "hideButton关系")

可以看到，仅从此处对比，并不能看出jQ的方便，因为毕竟功能太小，此处先引出jQ是如何获取元素的，引用一张来自w3school的图：
![](https://file-1305436646.file.myqcloud.com/blog/2017-10-18/SelectProgram.png "hideButton关系")

即可从图上看出id，class，标签等的获取语法

## 二、jQ设置内容

jQ库与js原生代码的比较如下：
![](https://file-1305436646.file.myqcloud.com/blog/2017-10-18/settingContent.png "settingContent关系")

输出结果如下：
![](https://file-1305436646.file.myqcloud.com/blog/2017-10-18/settingContentOut.png "settingContent输出")

此处依然不能体现出jQ的方便之处，还是因为功能并不复杂，但是之后的比较，会发现jQ的方便之处

## 三、jQ添加删除元素

jQ库与js原生代码的比较如下：
![](https://file-1305436646.file.myqcloud.com/blog/2017-10-18/addElement.png "addElement")

输出结果如下：
![](https://file-1305436646.file.myqcloud.com/blog/2017-10-18/addElementOut.png "addElementOut")

注意js原生代码部分两种方法的不同，第一种方法是与左侧jQ库写出的效果相同，第二种则是直接创建p标签进行追加文本

从此处其实已经可以看出jQ的方便之处，降低了代码复杂度，使代码更加简洁。

## 四、jQ功能简介

jQ库功能可以说是很强大，把js代码实现某些功能变得简单了很多。jQ几个主要实现的功能包括以下几个方面：

* 对DOM的操作，比如上面介绍的几种
* jQ的动画效果，比如淡入淡出效果、滑动效果
* Ajax的功能，使Ajax的写法变得简单

当然功能应该不止这些，这只是我目前所了解的功能的一个概括，其他具体功能可从网上搜索。

## 五、结语

总感觉目前写的这几篇博客内容太水了，技术深度不够，只是一些很基本的内容，没有什么深入的剖析。仅仅是对所学内容一个简单的使用而已，希望能随着技术的提高，逐渐写一些有深度的文章。