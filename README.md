# blog-solitude

基于 Hexo 的博客项目，主题通过 Git submodule 挂载在 `themes/solitude`。

## 环境要求

- Node.js 14+
- Yarn 或 npm
- Git

## 拉取仓库

首次克隆时，建议直接连同子仓库一起拉取：

```bash
git clone --recursive <仓库地址>
cd blog-solitude
```

如果主仓库已经拉下来了，再初始化并同步子仓库：

```bash
git submodule update --init --recursive
```

如果后续主仓库更新了子仓库指针，执行：

```bash
git pull
git submodule update --init --recursive
```

## 安装依赖

使用 Yarn：

```bash
yarn install
```

或使用 npm：

```bash
npm install
```

## 本地启动

启动本地开发服务：

```bash
yarn server
```

或：

```bash
npm run server
```

默认是 Hexo 本地服务，可在浏览器访问 `http://localhost:4000`。

## 常用命令

清理缓存文件：

```bash
yarn clean
```

生成静态文件：

```bash
yarn build
```

部署：

```bash
yarn deploy
```

如果使用 npm，对应命令分别是：

```bash
npm run clean
npm run build
npm run deploy
```

## 目录说明

- `source/`: 博客内容与静态资源
- `themes/solitude/`: 主题子仓库
- `_config.yml`: Hexo 主配置
- `_config.solitude.yml`: 主题配置
