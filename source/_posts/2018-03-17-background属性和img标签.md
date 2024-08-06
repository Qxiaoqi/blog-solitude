---
title: background属性和img标签
date: 2018-03-17 16:10:54
toc: true
categories:
- 前端
tags:
- CSS
cover: https://file-1305436646.file.myqcloud.com/blog/banner/gui-dang.webp

---

background这个属性可以说是很常见了，基本上是使用的非常频繁的一个属性，但是真说起来background属性，我又不能说是十分的清楚。本来说准备整理一下background属性，但是在整理的时候又发现了许多不清楚的地方，比如说和img区别？那么先把问题一个一个罗列下来，逐个解决。

在此，我要更正一个错误。那就是，原来我认为能用background-image解决的就不用img标签(似乎img用不到了？)，但是详细了解之后认为这是不正确的，区别将在问题一进行介绍。

<!--more-->

## 问题

1.background-image和img的区别？

2.background-size各个属性功能？

3.什么时候使用background什么时候使用img？

4.使用background-image的时候可以使其自适应屏幕宽度（比如说做轮播图的时候，可以指定宽度为width:100%，然后再指定一个height，使用background-size来让其充满），但是若使用img的话如何来达到这种效果。在我的印象中，img若指定width:100%的话height也会相应的变化，意思就是成比例变化，如果说同时指定width和height的话，若比例不对，图像会变形（拉伸、压缩之类的）。那么这种情况下，如何适应用户的不同屏幕？这个问题也是整理这篇文章的最大原因。


## 解答

### 问题一：background-image和img的区别 && 问题三：什么时候使用background什么时候使用img？

边整理区别的时候，再stack Overflow上看到了一个整理的很好的答案，感觉开启了新世界的大门。在此，我要推荐一下stack Overflow这个网站，除了全是英语阅读比较困难外，基本上很多问题都能在上面找到。果然对程序员来说stack Overflow和github两个网站就能解决大部分问题。比搜索引擎查的答案更加的全面，在此引用一下答案

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/answer.png" alt="answer">

上面的答案可以说是很全面了，那么大概翻译一下，提取几个要点，如下

##### 使用img的情况

* 当你想让打印页面时，你想让图片被默认包含在你的页面上，请使用img标签

* 当图像带有重要语义的时候，比如警告图标，请使用img标签（带有alt文本）因为这样的话即时因为一些问题而使图片没有加载出来，但是由于有alt属性，可以让文本代替图片，以至于重要信息不会因为图片无法加载而丢失。想象一下，如果使用background-image的话，倘若没法加载图片将会怎么样，也许就会错过一些重要的信息

* 如果图像是内容的一部分，最好使用img+alt 

* 如果你想让图像成比例放大或缩小，请使用img标签。background-image只能制定宽和高，虽然可以指定background-size，但依然不是成比例的，必然有一部分内容会缺失。

* 使用img而不是背景图像可以显著提高动画在后台的性能【这句话暂时并未理解】

* 当你想你的图片能被搜索引擎搜索到时，或者说更便于做SEO【答案来自其他人】

##### 使用backgroung-image的情况

* 如果图像不是内容的一部分，请使用CSS背景图像

* 使用CSS背景图像做图像替换的文本

* 如果你需要提高下载时间，就像CSS sprites 【css sprites的一个特性是似乎能让所有图片合成一张加载，这样能改善下载时间，为什么呢，学过数字图像处理可以知道，每个图像的颜色表都不一样，每一张图都需要一个颜色表的话自然不如只有一个颜色表占用空间少。具体css sprites内容还不太清楚】

* 使用backgroung-size，以拉伸背景图像填充整个窗口。

#### 总结

那么整理一下可以知道了，当使用不包含内容的图像，比如说背景的时候，那么完全可以使用background-image来让工作变得更加简单。但是倘若包含重要内容，应该使用img来作为一个HTML标签存在其中。

### 问题二：background-size各个属性功能？

引自MDN

> background 是CSS简写属性，用来集中设置各种背景属性。background 可以用来设置一个或多个属性:background-color, background-image, background-position, background-repeat, background-size, background-attachment。

background初始值：

> background-image: none
  background-position: 0% 0%
  background-size: auto auto
  background-repeat: repeat
  background-origin: padding-box
  background-clip: border-box
  background-attachment: scroll
  background-color: transparent

* `background-color`用于设置背景色

* `background-image`用于引入图片

* `background-position`规定背景图像位置

* `background-repeat`规定是否平铺

* `background-size`规定图像尺寸

* `background-attachment`设置背景图像是否固定或者随着页面的其余部分滚动

##### background-position

测试图700px,700px

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/demo.jpg" height="200px">

```html
<!DOCTYPE html>
<html>
<head>
  <title>background-position</title>
  <style type="text/css">
    .test {
      width: 600px;
      height: 600px;
      background-image: url("demo.jpg");
      //background-position: 0% center;
    }
  </style>
</head>
<body>
  <div class="test"></div>
</body>
</html>
```

将position注释的时候，可以看到处于默认状态0% 0%的状态，那么注释之后原图变成了

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab1-1.jpg" height="200px">

可以看到左边框的黑线仍能显示，而上边框的黑线已经没有了，那么得出结论，第一个值可以理解成左右位置，而第二个则可以理解成上下位置，center则代表上下居中。其他同理。

##### background-repeat

当图片宽或者高小于css指定的宽或者高时，空白的部分将被平铺，如图

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab2-1.jpg" height="200px">

```css 
.test {
      width: 730px;
      height: 730px;
      background-image: url("demo.jpg");
      background-color: #000000;
      /*background-position: 0% center;*/
      background-repeat: repeat;
    }
```

当设置成no-repeat时，可以看到有黑色背景透出，则可看出效果。

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab2-2.jpg" height="200px">

```css 
background-repeat: no-repeat;
```


##### background-size

做一个测试，来说明具体工作情况

```css 
.test {
      width: 400px;
      height: 300px;
      background-image: url("demo.jpg");
      background-color: #000000;
    }
```


<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab3-1.jpg" height="200px">

可以看到不设置background-size时显示不全

1.当都设置成100%时

```css 
.test {
      width: 400px;
      height: 300px;
      background-image: url("demo.jpg");
      background-color: #000000;
      background-repeat: no-repeat;
      background-size: 100% 100%;
    }
```

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab3-2.jpg" height="200px">

可以看到长和宽都被拉伸到了最大限度

2.当第一个设置成100%时

```css 
background-size: 100% auto;
```

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab3-3.jpg" height="200px">

可以看到宽被拉到100%

3.当第二个设置成100%时

```css 
background-size: auto 100%;
```

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab3-4.jpg" height="200px">

可以看到高被拉到了100%

4.设置成cover

> 把背景图像扩展至足够大，以使背景图像完全覆盖背景区域。背景图像的某些部分也许无法显示在背景定位区域中。

```css 
.test {
      width: 400px;
      height: 300px;
      background-image: url("demo.jpg");
      background-color: #000000;
      /*background-position: 0% center;*/
      background-repeat: no-repeat;
      background-size: cover;
    }
```

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab3-5.jpg" height="200px">

改变width和height

```css
.test {
      width: 300px;
      height: 400px;
      background-image: url("demo.jpg");
      background-color: #000000;
      /*background-position: 0% center;*/
      background-repeat: no-repeat;
      background-size: cover;
    }
```

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab3-6.jpg" height="200px">

可以看到结果不一样，即解释了什么叫做使背景完全覆盖背景区域,就是说将宽或者高大的哪一个拉到最大

5.设置成contain 

> 把图像图像扩展至最大尺寸，以使其宽度和高度完全适应内容区域。

测试方法同上，第一个设置`width:400px;height:300px`第二个设置`width:300px;height:400px`，结果如下，第一个

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab3-7.jpg" height="200px">

第二个

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-17/Lab3-8.jpg" height="200px">

则可以充分看出确实是高或者宽完全适应背景区域


##### background-attachment 

> scroll  默认值。背景图像会随着页面其余部分的滚动而移动。
> fixed  当页面的其余部分滚动时，背景图像不会移动。
> inherit  规定应该从父元素继承 background-attachment 属性的设置。

默认是会滚动的，当设置成fixed的时候，背景就会固定不动，可以借此做出比较炫酷的效果。


### 问题四：关于轮播图

关于这个问题，我想了想，倘若使用img标签的话（通过上面整理img应该更适合做轮播图，因为它在页面中可以说是比较重要的元素之一），那么就要从图片的大小上下手了，图片必须有一定的规范。首先轮播图的每个图片大小必须相同（或者说宽高比例相同），而且比例要适应浏览器，防止宽高比例不均的时候发生页面过大或过小的情况。我想这可能都是一些设计师的规范，而我对设计的领域了解并不深入，也许我想的是对的，也许是错的。但是这是我目前能想到的唯一解决方案。

使用background确实方便，但是为了网站的规范，我觉得有必要使用img标签。

## 结语

没想到竟然整理了这么多，background元素属性虽然看起来简单，但整理之前真让我详细的说我并不能说出来十分确切的内容，整理之后觉得对这方面的内容了解得更加深入的许多。
