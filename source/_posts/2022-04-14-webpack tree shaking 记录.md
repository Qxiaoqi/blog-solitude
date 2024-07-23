---
title: Webpack tree shaking 记录
date: 2022-04-14 15:38:25
toc: true
categories:
  - 前端
tags:
  - Webpack
cover: https://iph.href.lu/500x250
---

# Webpack tree shaking 记录

## 引例

主要记录一下中间遇到的一个问题，有如下代码，请问在最后打包的结果是什么？会包含 utils 和 infrad 的内容么？

场景：

- webpack 构建
- production 环境
- TerserPlugin 压缩

```js
import utils from "./utils";
import ReactECharts from "echarts-for-react"; // 任意一个第三方库（package.json 没有指定 sideEffects）
```

```js
// ./utils.js
export function square(x) {
  return x * x;
}

export function cube(x) {
  return x * x * x;
}

const exportUtil = {
  square: square,
  cube: cube,
};

export default exportUtil;
```

## 结果

包含 `ReactECharts`，但不包含 `utils`

## 原因

https://webpack.docschina.org/guides/tree-shaking/#mark-the-file-as-side-effect-free

> 在一个纯粹的 ESM 模块世界中，很容易识别出哪些文件有副作用。然而，我们的项目无法达到这种纯度，所以，此时有必要提示 webpack compiler 哪些代码是“纯粹部分”。

> 通过 package.json 的 "sideEffects" 属性，来实现这种方式。

> "side effect(副作用)" 的定义是，在导入时会执行特殊行为的代码，而不是仅仅暴露一个 export 或多个 export。举例说明，例如 polyfill，它影响全局作用域，并且通常不提供 export。

因此，自己开发的 `utils` 文件中，`webpack` 能自动分析出是否使用，最终在压缩阶段进行 `tree shaking`。

但是对于引入的第三方库，没办法判断解析（可能有其他插件做了改进？），因此会把 `echarts-for-react` 整个打包进来，如果想不打包进来，需要在 `echarts-for-react` 的 `package.json` 中声明 `sideEffects: false`（但要确保其确实没有副作用），这样在没有使用 `echarts-for-react` 的场景下，就不会将 `echarts-for-react` 打包进来。

## 其他

可以考虑更深入的解析 AST 分析作用域，来达到更加精确的 tree shaking ？