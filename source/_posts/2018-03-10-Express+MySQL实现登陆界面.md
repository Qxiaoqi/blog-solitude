---
title: Express+MySQL实现登陆界面
date: 2018-03-10 16:49:58
toc: true
categories:
- 项目
tags:
- 项目
- Express
- MySQL
cover: https://iph.href.lu/500x250

---

最近连着整理了好几篇博客，这些都是前一段时间做的东西，希望拿出来整理一下，加深印象。

之前看了一些nodejs，然后希望做出来点东西来实践，然后刚好在掘金上看到一个很棒的登陆效果，于是把他复现的同时也顺手做了个和后端交互的部分，就是注册账号，和登陆账号这么个简单的功能。然后发现express框架真的很好用，nodejs这块就采用了express框架来写，数据库用了Mysql。

<!--more-->

## 页面展示

感觉还是有点小炫酷的，登陆页面是一个动态效果，一个方块气泡上升的效果。

登陆界面
<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-10/display_1.jpg" height="300px">

注册界面
<img src="https://file-1305436646.file.myqcloud.com/blog/2018-3-10/display_2.jpg" height="300px">

## 技术细节

### 前端效果部分

#### 气泡效果

主要就是一层一层的往上压，使用z-index来控制谁在上面，该效果则是气泡在背景上面。

```html
<ul class="bg-bubbles">
  <li></li>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
  <li></li>
</ul>
```

css代码选取部分，其他的可以自己定义
```css 
ul li {
  position: absolute;
  z-index: 5;
  list-style: none;
  position: absolute;
  bottom: -160px;
  width: 40px;
  height: 40px;
  background-color: rgba(255, 255, 255, 0.15);
  animation: square 25s infinite;    //依次为动画名称，时间，播放次数（无限）
  transition-timing-function: linear;  //速度曲线（从头到尾相同）
  &:nth-child(1) {
    left: 10%;
  }
  &:nth-child(2) {
    left: 20%;
    width: 90px;
    height: 90px;
    animation-delay: 2s;
    animation-duration: 17s;
  }
  //以上自己定义


  //动画执行
  @keyframes square {
    0% {
      transform: translateY(-700px);
    }
        100% {
          transform: translateY(-700px) rotate(600deg);
        }
  }
```

上面代码使用less写的，less语法很简单，相比于sass应该来说语法少了很多，可以使用Sublime Text的插件来实现自动编译成css的效果，或者也可以使用webpack，gulp等自动胡工具通过编写任务来实现编译。

* 背景使用`background-image: linear-gradient( 135deg, #FFF6B7 10%, #F6416C 100%);`来分别控制颜色渐变方向，两个颜色

* 气泡的上升是通过`translateY()`来控制的

* 气泡的旋转是通过`rotate()`来控制的

* 但是有一处不知道为什么，气泡为什么会在最上面疑似停留一会才消失的效果，倘若是因为`transition-timing-function: `控制速度曲线的原因的话，那么此处设置的是`linear`应该是一直平均速度运行，为什么在最后会很慢呢。暂未解决

#### 弹窗效果

原理很简单，当用户点击注册新账号的时候会调用一个函数，改函数会将一个之前写好的并用display:none隐藏的div块重新改为display:block使之重新显现。同时也可以在之前写好一个大的div并用z-index控制层叠位置来使整个背景遮罩变暗

```html 
//背景遮罩
<div class="overlay"></div>

//注册窗口
<div class="register-box">
  <div class="close-box"></div>
  <div class="register-content">
    <div class="register-username">
      <input type="text" placeholder="请输入用户名" id="register-username">
    </div>
    <div class="register-password">
      <input type="password" placeholder="请输入密码" id="register-password">
    </div>
    <div  class="register-repassword">
      <input type="password" placeholder="请再次输入密码" id="register-repassword">
    </div>
    <div class="register-button">
      <button>立即注册</button>
    </div>
  </div>
</div>
```

```js 
var showRegister = function() {
  console.log("s");
  $(".overlay").css("display","block");
  $(".register-box").css("display","block");
}

var closeRegister = function() {
  $(".overlay").css("display","none");
  $(".register-box").css("display","none");
}
```



### express框架部分

#### 服务器运行

首先，这一部分要想跑起来的代码

```js
var server = app.listen(8888, function () {
 
  var host = server.address().address
  var port = server.address().port
 
  console.log("应用实例，访问地址为 http://%s:%s", host, port)
 
})
```

端口号为8888，同时输出对应访问地址

#### 设置路由

然后设置相应路由，即访问不同地址的时候返回给前端的内容不同


```js
app.get('/', function (req, res) {
  res.sendFile(__dirname + "/public/" + "index.html");
})

app.post('/process_login',function (req, res) {
 
   console.log(req.body.username);
   console.log(req.body.password);
   if(req.body.username === "xiaoqi" && req.body.password === "1111") {
       res.end(JSON.stringify(dataSuccess));
   } else {
        res.end(JSON.stringify(dataError));
   }   
})

//调用与数据库交互的方法
app.post('/process_register',function (req, res) {
 
   console.log(req.body.username);
   console.log(req.body.password);
   userDao.add(req, res);
     
})
```

* sendFile方法返回一个文件

* get方法时要返回数据给前端需要使用req.query来获取URL的查询参数串 

* post方法使用req.body获得请求主体 

* 然后返回给前端的要是JSON格式转成字符串的格式

* 此处还未使用mysql，所以假定了一个账号和密码方便此时的测试


#### 设置静态文件

当完成上述部分之后会发现相应的css与js内容并未配置到相应html文件中，在express提供了内置的中间件express.static来设置静态文件，例如将相应的文件都放在public目录下。

```js
app.use(express.static('public'));
```

使用一行简单的命令即可


#### json解析

当运行的时候会发现使用ajax提交的数据无法被解析，那么可以使用express中的一个bodyParser中间件来进行解析，简单的两行代码

```js 
// 添加json解析
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: false}));
```

* 使用之前要引入`var bodyParser = require('body-parser');`

### 登陆逻辑

这部分比较的没什么新的要整理的，主要就是ajax传值的时候记得用`JSON.stringify`把json格式转成字符串。


### mysql数据库部分

数据库部分为了使项目看起来更有层次，多添加了两个文件夹，conf用来放配置文件，dao用来放与数据库交互的部分

#### 创建mysql表

```sql
SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
--  Table structure for `websites`
-- ----------------------------
DROP TABLE IF EXISTS `users`;
CREATE TABLE `users` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(255) NOT NULL DEFAULT '' COMMENT '用户名',
  `password` varchar(255) NOT NULL DEFAULT '',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
--  Records of `websites`
-- ----------------------------
BEGIN;
INSERT INTO `users` VALUES ('1', 'xiaoqi', '1111');
COMMIT;

SET FOREIGN_KEY_CHECKS = 1;
```

只有三个值，分别是id，username和password非常的简单

#### conf中数据库配置文件

```js 
// conf/db,js
// MySQL数据库连接配置

module.exports = {
  mysql: {
    host: 'localhost',
    user: 'root',
    password: '1111',
    port: '3306',
    database: 'login_information'
  }
};
```

#### dao中数据库命令编写

```js
var user = {
  insert: 'INSERT INTO users(id, username, password) VALUE(0,?,?)',
  update: 'update users set username=?, password=? where id=?',
  delete: 'delete form users where id=?',
  queryById: 'select * from users where id=?',
  queryAll: 'select * from users'
};

module.exports = user;
```

#### dao中与数据库交互

```js 
//dao/userDao.js
//实现与Mysql交互
var mysql = require('mysql');
var $conf = require('../conf/db');
var $sql = require('./userSql');

// var connection = mysql.createConnection($conf.mysql);
var pool = mysql.createPool( $conf.mysql );
//向前端返回结果
var jsonWrite = function(res, ret) {
  if(typeof ret === 'undefined') {
    console.log("ret === undefined");
    console.log("ret =" + ret);
    res.end(JSON.stringify({
      status: '2',
      msg: '操作失败'
    }));
  } else {
    console.log("ret !== undefined");
    console.log(ret);
    res.end(JSON.stringify(ret));
  }
};

module.exports = {
  add: function(req, res) {
    console.log("add方法运行");
    pool.getConnection(function(err, connection) {
      if(err) {
        // console.log("pool报错");
        throw err;
      }
      var param = req.body;
      connection.query($sql.insert, [param.username, param.password], function(err, result) {
        if(result) {
          result = {
            status: 200,
            msg: '增加成功'
          };
        }

        //以json形式，把操作结果返回给前端
        jsonWrite(res, result);

        // 释放连接 
        connection.release();
      })
    })
  }
}
```

* 数据库这部分代码整合了网上的代码，并加上了我自己的内容，但是对于其中数据库部分的一些原理可能不是很了解，比如说pool连接池

* query方法就类似于直接在命令行中敲sql命令

可以在数据库中查询，每当注册新用户之后，确实发现数据库中新增了相应的数据，那么这个项目的核心已经基本完成了。


## 后记

整理这篇文章的时候大概已经是做完这个项目一周之后的事情了，法相其中的一些小细节已经忘了，只能整理一些大概的内容，感觉效果并不是非常的好，所以以后在做完项目的时候一定要第一时间整理，才能达到最好的效果。