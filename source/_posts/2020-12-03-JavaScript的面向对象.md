---
title: JavaScript 的面向对象
date: 2020-12-03 21:18:19
toc: true
categories:
  - 面试整理
tags:
  - JavaScript
cover: https://iph.href.lu/500x250
---

# JavaScript 的面向对象

本篇笔记结合专栏内容和其它内容做一个大致总结。主要涉及内容：对象，JS 的面向对象和其它语言的区别，继承相关。

## 对象相关

### 两类属性

数据属性：

- value：属性值
- writable：属性能否被赋值
- enumerable：可否被 for in 枚举
- configurable：可否被删除或改变特征值

访问器属性：

- getter：获取时调用
- setter：设置时调用
- enumerable：可否被 for in 枚举
- configurable：可否被删除或改变特征值

### 一些方法

- hasOwnProperty()：检测自有属性
- propertyIsEnumerable()：hasOwnProperty 的加强版，当自有属性的可枚举性是 true 时才返回 true
- Object.keys()：返回对象中可枚举的自有属性组成的名称数组
- Object.getOwnPropertyNames()：与 Object.keys()相比，返回的不仅仅时可枚举的属性
- Object.getOwnPropertyDescriptor()：获得某个对象自有属性的属性描述符
- Object.defineProperty()：设置属性的特性
- Object.getPrototypeOf()：查询其原型，ES5 之前使用 o.constructor
- isPrototypeOf()：可用来检测一个对象是否是另一个对象的原型，instanceof 用来检测是否是从构造函数实例的对象
- Object.prototype.toString.call()：可用来判断类型，基础类型都可判断比如 array，string，object，boolean，number，undefined，null

## JavaScript 和 Java 为代表的面向对象区别

- 以 Java 为代表的面向对象是基于类的面向对象
- JavaScript 是基于原型的面向对象

说到 JavaScript 的面向对象，就要提到 JavaScript 的诞生，因为一些原因，JavaScript 在推出之时就要求它去模仿 Java，因此产生了 new，this 等语言特性，使之看起来像 Java。

两者差别还是很大的

- 基于类的思想先有类，再用类去实例化一个对象
- 基于原型的思想则提倡关注实例的行为，然后才关心将这些对象划分到相似的原型对象，而不是分成类。

在我看来，不同语言有不同的特性，JavaScript 对象具有高度动态性，因为 JavaScript 赋予了使用者在运行时为对象添加状态和行为的能力，比如创建一个对象后，再给它添加属性，这样毫无问题。没有必要刻意的去模拟基于类的一些方法，应该充分利用 JavaScript 的优秀之处。

## JavaScript 的继承

说面向对象就要说到继承。

JavaScript 的继承方法有很多，从最开始的使用`function`来模拟 Java 类的语法（如：new，prototype，constructor 等），到 ES5 的`Object.create()`的出现，再到现在 ES6 的`class`和`extends`语法糖。随着标准的不断提出，也趋向于逐渐完善的状态。

不管是哪种方法，归根结底都是基于原型的继承，JavaScript 的原型链是其中的关键所在。

### 继承的状态

继承有三个重要的状态，脑海中要有一张图，分别是：原型对象，构造函数，对象

- 原型对象 constructor -> 构造函数
- 构造函数 prototype -> 原型对象
- 构造函数 new -> 实例
- 实例 \_\_proto\_\_ -> 原型对象

### 继承的方法

实现方式可在 demo 里找到

七种方式：

- 原型链继承
  - 实现：new 父类 赋给 子类 prototype
  - 缺点：实例对引用类型（如数组）的改动，会导致所有实例改动，而且不可向父类传参
- 构造函数式继承
  - 实现：子类 call 父类构造函数
  - 缺点：无法实现复用，所有子类有父类实例的副本，影响性能
- 组合式继承
  - 实现：上面两种综合使用
  - 缺点：创建实例的时候，原型中会有两份相同的属性（可用 寄生组合方式 改进，即 Object.create）
- 原型式继承
  - 实现：对象 Object.create 创建
  - 缺点：无法传递参数，有篡改可能
- 寄生式继承
  - 是一种思路，可以和组合方式组合
  - 缺点：同原型式继承
- 寄生组合式继承
  - 实现：在组合式继承的基础上改动，即将 new 父类的部分，改成 Object.create(父类.prototype)。原因，new 会执行目标函数，导致多创建一层，而 Object.create()不会执行，所以少一层。
  - 目前最为完善的方法
- 混入方式继承多个对象
  - 实现：Object.assign()会将其它原型上的函数拷贝到目标原型上，所以可以继承多个对象

备注：Object.create()是 ES5，原理是创建一个空函数，将传入的参数绑定到空函数的 prototype 上，然后返回 new f() 实例

### 继承总结

其实上面说的这么多方式，其实就三种思路

- 第一种是`function`来模拟，该种方法更像 Java 风格的类接口来操纵，非常的别扭
- 第二种即 ES5 的`Object.create()`直接创建（Object.create 可用其他方式模拟），这种方式在我看来更加符合基于原型的面向对象
- 第三种是 ES6 的`class extends`这种更符合工程中使用

## 代码示例

### 1. 原型链继承

```js
// 原型链继承
// 即重写原型对象
function SuperType() {
  this.x = "1";
  this.colors = ["red", "blue"];
  console.log("Super执行");
}

SuperType.prototype.getSuperValue = function() {
  return this.x;
}

function SubType() {
  this.y = "2";
  console.log("Sub执行");
}


SubType.prototype = new SuperType();
// SubType.prototype = Object.create(SuperType.prototype);
SubType.prototype.constructor = SubType;

// 注意此处，需要在将父类赋值给prototype后定义
SubType.prototype.getSubValue = function() {
  return this.y;
}

let test1 = new SubType();
let test2 = new SubType();
let test3 = Object.create(SubType.prototype);

console.log("测试：", test3 instanceof SubType);
// console.log("测试：", );

// 非引用则不会
test1.x = "#";
// 多个实例对引用类型的改动，会反映到整个链
// 引用传递（Array，Function，Object）
// test1.colors.push("black");
// 重新赋值，会创建新的引用地址
// test1.colors = ["111"];

console.log(test1.x);
console.log(test2.x);
console.log(test3.x);


console.log(test1.colors);
console.log(test2.colors);
console.log(test3.colors);
```

### 2. 借用构造函数继承

```js
// 借用父类构造函数实现，复制父类实例给子类（不使用原型）
function SuperType() {
  this.x = "1";
  this.colors = ["red", "blue"];
  console.log("Super执行");
}

SuperType.prototype.getSuperValue = function() {
  return this.x;
}

function SubType() {
  SuperType.call(this);
  this.y = "2";
  console.log("Sub执行");
}

SubType.prototype.getSubValue = function() {
  return this.y;
}

let test1 = new SubType();
let test2 = new SubType();


// 改动一个子实例引用并不会引起其它实例的改变
// 因为不同子实例是不同的副本
test1.colors.push("black");


console.log(test1.colors);
console.log(test2.colors);
```

### 3. 组合继承

```js
// 组合继承
function SuperType(x) {
  this.x = x;
  this.colors = ["red", "blue"];
  console.log("Super执行");
}

SuperType.prototype.getSuperValue = function() {
  return this.x;
}

function SubType(x, y) {
  SuperType.call(this, x);
  this.y = y;
  console.log("Sub执行");
}

SubType.prototype = new SuperType();
// SubType.prototype = Object.create(SuperType.prototype);
// 重写constructor属性，指向自己的构造函数
SubType.prototype.constructor = SubType;

SubType.prototype.getSubValue = function() {
  return this.y;
}

let test1 = new SubType(1, 2);
let test2 = new SubType(3, 4);

test1.colors.push("black");


console.log(test1.colors);
console.log(test2.colors);

console.log(test1.x);
console.log(test2.x);

console.log(test1.y);
console.log(test2.y);
```

### 4. 原型式继承

```js
let Person = {
  name: "xiao",
  age: 20
}

function inheritObject(proto) {
  function f() {}
  f.prototype = proto;
  return new f();
}

let person1 = inheritObject(Person);
let person2 = inheritObject(Person);

console.log("测试：", Person.isPrototypeOf(person1));

Person.age = 10;

console.log(person1.age);
console.log(person2.age);
```

### 5. 寄生组合式继承

```js
function inheritPrototype(SuperType, SubType) {
  SubType.prototype = Object.create(SuperType.prototype);
  SubType.prototype.constructor = SubType;
}

function SuperType(name) {
  this.name = name;
  this.colors = ["blue", "red"];
  console.log("Super执行");
}

SuperType.prototype.getName = function() {
  console.log("getName:", this.name);
}

function SubType(name, age) {
  SuperType.call(this, name);
  this.age = age;
  console.log("Sub执行");
}

inheritPrototype(SuperType, SubType);

SubType.prototype.getAge = function() {
  console.log("age:", this.age);
}

var test1 = new SubType("xiao", 20);
var test2 = new SubType("da", 30);


test1.colors.push("black");

console.log(test1.age);
console.log(test2.age);

console.log(test1.colors);
console.log(test2.colors);
```

### 6. ES6

```js
class Point {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  toString() {
    return this.x + this.y;
  }
}

class ColorPoint extends Point {
  constructor(x, y, color) {
    super(x, y);
    this.color = color;
  }

  toString() {
    return this.color + " " + super.toString();
  }
}

let p1 = new Point(1, 2);
let p2 = new ColorPoint(1, 3, "blue");

console.log(p1.toString());
console.log(p2.toString());
```