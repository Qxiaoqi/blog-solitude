---
title: Vue-Router 整理
date: 2020-12-07 16:18:19
toc: true
categories:
  - 面试整理
tags:
  - Vue
cover: https://file-1305436646.file.myqcloud.com/blog/banner/gui-dang.webp
---

# Vue-Router

## 懒加载

箭头函数 + import()，现在一般都用这个

```js
const List = () => import("@/components/list.vue")
const router = new VueRouter({
  routes: [{ path: "/list", component: List }],
})
```

## hash 模式 history 模式

### hash 模式

是开发中默认的模式，它的 URL 带着一个#，例如：www.abc.com/#/vue，它的hash值就是#/vue

原理：使用 window.onhashchange() 事件

### history 模式

相比 hash 模式 url 会更加的整洁，需要服务端配合配置 location

原理：使用 history API pushState() 和 replaceState() 来修改历史记录

### history 优势

1. url 更整洁
2. pushState 的参数可以比较复杂，比如对象，而 hash 模式只能是字符串

## $route 和$router 的区别

【$route】 是“路由对象”，包括 path，params，hash，query，fullPath，matched，name 等路由信息参数
【$router】 是“路由实例”对象包括了路由的跳转方法 `this.$router.push()`，钩子函数等。

## vue-router 和 window.location.href 区别

vue-router 是使用 history API 或者 hashchange 来做的，不会触发浏览器刷新，而 location.href 是会触发浏览器刷新的

history 模式通过修改浏览器历史记录，做到 url 改变但是不刷新页面的效果。

## 前端路由的理解

最早之前大概是一个 url 对应一个页面这种。

后来出现了 ajax，能允许在不刷新页面的情况下发起请求，局部的更新内容。但是这有一种问题，没办法记录当前的位置，比如刷新一下之后，就会回到最初的状态。

前端路由就是为了解决这个问题，记录用户的通过一些操作之后触发的前进后退等等内容，记录到浏览器历史记录中，这样刷新之后也会定位到当前的内容，对于 Vue 来讲主要就是两种模式，hash 模式和 history 模式。一个是通过 hashchange，一个是使用 histrory API 提供的 pushState 等方法去修改浏览器历史记录。
