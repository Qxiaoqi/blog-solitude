---
title: CSS-iScroll实现水平滚动实例
date: 2018-09-07 19:46:11
toc: true
categories:
- 前端
tags:
- CSS
cover: https://file-1305436646.file.myqcloud.com/blog/banner/gui-dang.webp

---

## 前言

最近在看《CSS世界》这本书，这本书其中的很多小技巧都让我有种眼前一亮的感觉，而且还有很多我之前不了解的一些语法，那么我会抽空将这本书上的一些小技巧，或者一些例子实现一下，加深印象。其中也会有一些自己的思考，以及学习这些知识时我发现的其它技术内容。

<!--more-->

## iScroll

iScroll这是一个更多应用于移动设备的开发上的工具，能实现滚动效果、滑动等很多的效果，之前对这个工具并不了解。

## 具体实现

<img src="https://file-1305436646.file.myqcloud.com/blog/2018-9-7/demo.jpg" height="300px">

#### HTML

```html
<div id="wrap" class="wrap">
  <ul>
    <li><img src="test.jpg"></li>
    <li><img src="test.jpg"></li>
    <li><img src="test.jpg"></li>
    <li><img src="test.jpg"></li>
    <li><img src="test.jpg"></li>
  </ul>
</div>
```

#### CSS

```css
.wrap {
  margin: 200px;
  width: 300px; height: 200px;
  position: relative;
  overflow: hidden;
}
.wrap > ul {
  position: absolute;
  white-space: nowrap;
}
.wrap li {
  display: inline-block;
}
.wrap li img {
  height: 192px;
}
```
#### JS

```js
// 这里需要引入iscroll.js
new IScroll('#wrap', {
  scrollbars: true,
  scrollX: true,
  scrollY: false,
});
```
如果DOM还没有渲染完就执行iScroll初始化，会没有效果

> 如果你有一个复杂的DOM结构，最好在onload事件之后适当的延迟，再去初始化iScroll。最好给浏览器100或者200毫秒的间隙再去初始化iScroll

> iScroll使用的是querySelector 而不是 querySelectorAll，所以iScroll只会作用到选择器选中元素的第一个

## iScroll所能实现的一些功能

* `scrollTo(x, y, time, easing)`滚动到任意位置
* `scrollBy(x, y, time, easing)`从当前位置相对滚动
* `scrollToElement(el, time, offsetX, offsetY, easing)`滚动到特定位置
* `goToPage(x, y, time, easing)`滚动到想要滚动的页面数（需在参数配置里设置snap）

当然还有一些别的功能，这里目前没有需求，所以就不列举其它的功能了。

## 总结

那么从这个工具上来看，可以说是非常适合移动端的手指滑动了，而且动画效果非常的流畅。除此之外，想到了之前做的一个问卷平台，问卷平台的右侧有一个跟随题目生成位置不断移动的窗口，那么这里就可以通过iScroll来实现这种效果。


