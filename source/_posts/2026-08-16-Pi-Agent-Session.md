---
title: Pi 源码解析（六）：Session 会话树与持久化
date: 2026-08-16 21:11:00
toc: true
recommend: true
categories:
- 大模型
- Pi
tags: 
- 大模型
- Pi
- 源码解析
cover: https://file-1305436646.cos.ap-nanjing.myqcloud.com/blog/banner/2026-08-16.webp
---

# Pi 源码解析（六）：Session 会话树与持久化

> Pi Agent Core 源码系列（6/8）｜上一篇：{% post_link 2026-08-15-Pi-AgentHarness "AgentHarness 的状态和会话" %}｜下一篇：{% post_link 2026-08-17-Pi-Context-Compaction "上下文压缩与分支记忆" %}

对一个简单聊天应用来说，Session 似乎只是一个不断增长的消息数组：用户发一条消息，模型回一条消息，再把它们依次追加到历史中。但 Coding Agent 的会话并不总是沿一条直线前进。用户可能回到旧消息重新提问、重新生成回答、比较不同实现方案，或者在一个失败分支之后退回共同祖先继续探索。

Pi 把完整历史保存为一棵 append-only 会话树，再把当前活动分支转成模型需要的线性消息。这样既能保留回退和重新生成的历史，又不需要让模型理解树结构。

## Session 位于哪一层

Pi 将会话的业务语义和具体存储方式分开：

{% mermaid %}
flowchart TD
    H[AgentHarness<br/>运行编排、事件、配置] --> S[Session<br/>会话树领域层]
    S --> I[SessionStorage<br/>统一存储契约]
    I --> M[InMemorySessionStorage<br/>测试与临时会话]
    I --> J[JsonlSessionStorage<br/>本地持久化]
    S --> C[SessionContext<br/>模型消息与分支状态]
{% endmermaid %}

三层分工如下：

- `AgentHarness` 决定何时执行模型、工具、压缩和分支跳转；
- `Session` 解释各种会话条目的含义，并构建当前上下文；
- `SessionStorage` 负责追加、查询和恢复树；
- Memory 与 JSONL 只是同一存储契约的不同实现。

这使“会话是什么”和“会话存在哪里”成为两个独立问题。

## 为什么不是消息数组

假设原始对话是：

```text
用户问题 1 → 模型回答 1 → 用户问题 2 → 模型回答 2
```

用户现在回到“模型回答 1”，修改问题并重新探索。如果使用数组，要么删除原来的后续消息，要么复制公共历史形成第二份数组。前者丢失历史，后者造成重复和分支管理困难。

树结构可以直接表达这种关系：

{% mermaid %}
flowchart TD
    U1[用户问题 1] --> A1[模型回答 1]
    A1 --> U2[用户问题 2]
    U2 --> A2[模型回答 2<br/>原分支]
    A1 --> U3[修改后的问题 2]
    U3 --> A3[新的模型回答<br/>当前 leaf]

    classDef active fill:#dff7e8,stroke:#26834a,color:#123a24,stroke-width:2px;
    class U1,A1,U3,A3 active;
{% endmermaid %}

完整 Session 保存整棵树，但当前模型只看到绿色路径：

```text
用户问题 1 → 模型回答 1 → 修改后的问题 2 → 新的模型回答
```

> 完整历史是树，当前上下文是树中从根到 leaf 的一条路径。

## Session Tree Entry 记录了什么

树中的节点不只有聊天消息。Pi 把与会话有关的变化都表示成 `SessionTreeEntry`，每个条目拥有自己的 ID、父节点、时间和类型。

这些条目大致分成四类：

| 类别 | 典型条目 | 作用 |
|---|---|---|
| 对话内容 | message、custom message | 保存用户、模型和应用消息 |
| 运行状态 | model change、thinking level change、active tools change | 记录当前分支的配置变化 |
| 上下文管理 | compaction、branch summary | 控制长会话和分支信息 |
| Session 管理 | label、session info、leaf、custom entry | 保存标签、名称、游标和应用数据 |

这种设计接近 Event Sourcing：系统通常不原地修改过去状态，而是追加一条“发生了什么”的记录，再通过重放当前分支恢复最终状态。

{% mermaid %}
flowchart LR
    E1[message] --> E2[assistant message]
    E2 --> E3[model_change<br/>切换到 GPT]
    E3 --> E4[thinking_level_change<br/>切换到 high]
    E4 --> E5[user message]
{% endmermaid %}

模型切换不是 Session 对象上一个孤立的全局字段，而是当前分支历史的一部分。不同分支因此可以自然拥有不同模型、思考级别和启用工具。

## 活动 Leaf：当前会话在哪里

树本身包含所有探索方向，但应用还需要知道用户当前位于哪个节点。这个节点就是活动 leaf。

{% mermaid %}
flowchart TD
    R[根] --> A[公共历史]
    A --> B1[分支 A]
    B1 --> B2[分支 A 的最新消息]
    A --> C1[分支 B]
    C1 --> C2[分支 B 的最新消息<br/>当前 leaf]

    classDef leaf fill:#fff2cc,stroke:#b8860b,color:#4a3500,stroke-width:2px;
    class C2 leaf;
{% endmermaid %}

只要知道 `leafId`，存储层就能沿 `parentId` 向上回溯到根，再反转得到当前分支：

```text
leaf → parent → parent → root
                 ↓ 反转
root → ... → parent → leaf
```

从模型角度看，这仍然是普通的线性对话历史。

## LeafEntry：把“移动游标”也写进日志

如果 leaf 只是一个内存变量，用户回到旧节点后，一旦程序退出，这次移动就会丢失。重新读取所有普通条目时，系统只会把最后追加的消息误认为当前 leaf。

Pi 的做法是把 leaf 移动也保存为 append-only 控制记录。假设当前位于 `a2`，用户回到 `a1`：

```text
普通日志：u1 → a1 → u2 → a2
游标移动：追加 LeafEntry(targetId = a1)
```

完整时间线变成：

{% mermaid %}
flowchart LR
    U1[u1<br/>普通 entry] --> A1[a1<br/>普通 entry]
    A1 --> U2[u2<br/>普通 entry]
    U2 --> A2[a2<br/>普通 entry]
    A2 --> L1[leaf entry<br/>targetId = a1]
    L1 --> Cursor[最终活动 leaf = a1]
{% endmermaid %}

重启时顺序重放：

| 读到的条目 | 重放后的 leaf |
|---|---|
| u1 | u1 |
| a1 | a1 |
| u2 | u2 |
| a2 | a2 |
| leaf entry，targetId 为 a1 | a1 |

这里需要区分三个字段：

- `id` 是这条“移动记录”自己的 ID；
- `parentId` 表示移动前位于哪里；
- `targetId` 表示移动后要去哪里。

当前 leaf 是 `targetId`，不是 LeafEntry 自己的 ID。LeafEntry 也不会作为消息进入模型上下文，它只负责持久化游标变化。

移动到 `a1` 后再追加一条新消息，新消息的父节点就是 `a1`，于是自然形成新分支：

{% mermaid %}
flowchart TD
    U1[u1] --> A1[a1]
    A1 --> U2[u2]
    U2 --> A2[a2<br/>旧分支]
    A1 --> U3[u3<br/>移动后追加的新消息]
{% endmermaid %}

## 两种视图：树结构与追加日志

理解 LeafEntry 的关键，是同时保留两个视角。

### 对话树视角

描述节点之间的父子关系：

```text
u1
└── a1
    ├── u2
    │   └── a2
    └── u3
```

### 追加日志视角

描述真实发生过的操作顺序：

```text
u1
a1
u2
a2
leaf：移动到 a1
u3
```

树回答“历史有哪些分支”，日志回答“这些变化按什么顺序发生”。Pi 用同一组 Entry 同时支持这两个视角。

## 分支状态如何恢复

当前分支不仅决定模型看到哪些消息，也决定当前配置。Session 会沿 root-to-leaf 路径依次重放状态变化，后出现的配置覆盖前面的配置。

{% mermaid %}
flowchart LR
    Start[默认状态<br/>thinking = off] --> M1[assistant<br/>实际模型 Claude]
    M1 --> M2[model_change<br/>GPT]
    M2 --> T1[thinking_level_change<br/>high]
    T1 --> Tools[active_tools_change<br/>read + bash]
    Tools --> Result[当前分支状态<br/>GPT / high / read+bash]
{% endmermaid %}

如果从共同祖先进入另一个没有经过这些配置条目的分支，另一个分支不会错误继承它们。这是把配置变化放进树，而不是放进全局可变对象的重要价值。

状态恢复本身是一遍从根到 leaf 的顺序归约。后出现的配置覆盖旧值，assistant 消息也能恢复当时实际使用的模型：

```ts
function deriveSessionContextState(
  pathEntries: readonly SessionTreeEntry[],
): Omit<SessionContext, "messages"> {
  let thinkingLevel = "off";
  let model: { provider: string; modelId: string } | null = null;
  let activeToolNames: string[] | null = null;

  for (const entry of pathEntries) {
    if (entry.type === "thinking_level_change") {
      thinkingLevel = entry.thinkingLevel;
    } else if (entry.type === "model_change") {
      model = { provider: entry.provider, modelId: entry.modelId };
    } else if (entry.type === "message" && entry.message.role === "assistant") {
      // assistant 保存实际响应它的 Provider 和模型，可用于恢复运行状态。
      model = { provider: entry.message.provider, modelId: entry.message.model };
    } else if (entry.type === "active_tools_change") {
      activeToolNames = [...entry.activeToolNames];
    }
  }

  return { thinkingLevel, model, activeToolNames };
}
```

这段逻辑位于 [`session.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/session/session.ts)。它使用完整分支恢复控制状态，不依赖稍后经过 Compaction 处理的模型消息视图。

## 从完整分支到模型上下文

Session 中的条目并不都会发送给模型。构建上下文时，Pi 将“状态恢复”和“消息投影”分成两条路径：

{% mermaid %}
flowchart TD
    Branch[当前完整分支<br/>root → leaf]
    Branch --> State[状态归约]
    Branch --> View[上下文条目变换]

    State --> Model[当前模型]
    State --> Thinking[Thinking Level]
    State --> ActiveTools[Active Tools]

    View --> Projection[Entry → AgentMessage 投影]
    Projection --> Messages[模型可见消息]

    Model --> Context[SessionContext]
    Thinking --> Context
    ActiveTools --> Context
    Messages --> Context
{% endmermaid %}

普通消息直接进入上下文；模型变化、标签、leaf 等结构条目只影响 Session 行为，不直接暴露给模型；compaction 和 branch summary 则会转成专门的摘要消息。

这种分离避免了两个问题：

1. 模型不需要看到内部控制记录；
2. 控制状态不会因为消息裁剪而丢失。

## 摘要也是 Session Entry

Compaction 和 Branch Summary 都以新 Entry 的形式进入树，而不是覆盖或删除旧消息。前者改变当前分支的上下文投影，用摘要替代较早历史并保留近期原文；后者在切换位置时携带离开分支的经验。Session 只负责解释这些条目如何进入当前路径，Token 预算、切分边界和摘要生成由下一篇 {% post_link 2026-08-17-Pi-Context-Compaction "Context Compaction" %} 说明。

状态恢复仍基于完整 root-to-leaf 路径。即使早期消息不再出现在模型上下文中，其中记录的模型、thinking level 和 active tools 变化也不会丢失。

## 存储抽象：Memory 与 JSONL

`SessionStorage` 把树操作抽象为统一契约，主要包括：

- 追加 entry；
- 按 ID 查询节点；
- 获取当前 leaf；
- 持久化 leaf 移动；
- 获取 root-to-leaf 路径；
- 列出完整追加日志。

两个内置实现面向不同场景：

| 实现 | 数据位置 | 适用场景 | 主要特点 |
|---|---|---|---|
| InMemorySessionStorage | Array 与 Map | 测试、临时 Agent、一次性任务 | 简单快速，进程退出后默认丢失 |
| JsonlSessionStorage | 本地 JSONL 文件 | CLI 正式会话、本地恢复与审计 | 一行一个记录，适合 append-only 写入 |

{% mermaid %}
flowchart LR
    Session[Session 领域层] --> Contract[SessionStorage 契约]
    Contract --> Memory[Memory<br/>entries + byId + leafId]
    Contract --> Jsonl[JSONL<br/>header + entry lines]
    Contract -.可扩展.-> DB[数据库实现<br/>服务端并发场景]
{% endmermaid %}

Memory 实现中的路径恢复很直接：沿 `parentId` 回溯，再用 `unshift()` 保持根到 leaf 的顺序。

```ts
/** 沿 parentId 从 leaf 回溯到根，再反转成时间正序路径。 */
async getPathToRoot(leafId: string | null): Promise<SessionTreeEntry[]> {
  if (leafId === null) return [];

  const path: SessionTreeEntry[] = [];
  let current = this.byId.get(leafId);
  if (!current) throw new SessionError("not_found", `Entry ${leafId} not found`);

  while (current) {
    // unshift 使最终数组保持根 → leaf 顺序。
    path.unshift(current);
    if (!current.parentId) break;

    const parent = this.byId.get(current.parentId);
    if (!parent) throw new SessionError("invalid_session", `Entry ${current.parentId} not found`);
    current = parent;
  }
  return path;
}
```

JSONL 存储在加载文件后构建同样的 ID 索引，因此查询路径的算法不变，区别主要在追加写入和恢复过程。

JSONL 与 Session 的 append-only 模型非常匹配：新增消息、配置变化和 leaf 移动都只需要在文件末尾追加一行，不必读取并重写整个历史文件。

Memory 和 JSONL 的选择可以简化为：

```text
测试或一次性任务          → Memory
本地 CLI，需要恢复历史    → JSONL
多用户、多实例、需要事务  → 自定义数据库 SessionStorage
```

## 一次操作如何穿过 Session

把前面的概念串起来，一次典型分支操作如下：

{% mermaid %}
sequenceDiagram
    participant U as 用户
    participant H as AgentHarness
    participant S as Session
    participant ST as SessionStorage

    U->>H: 回到历史节点 a1
    H->>S: moveTo(a1)
    S->>ST: setLeafId(a1)
    ST->>ST: 追加 LeafEntry(targetId = a1)

    U->>H: 发送新问题
    H->>S: appendMessage(u3)
    S->>ST: appendEntry(parentId = a1)
    ST->>ST: leafId = u3

    H->>S: buildContext()
    S->>ST: getPathToRoot(u3)
    ST-->>S: root → a1 → u3
    S-->>H: SessionContext
{% endmermaid %}

在这个过程中，旧分支没有被删除，当前上下文也没有包含 LeafEntry。树负责保留全部历史，leaf 负责选择当前路径，Context 投影负责把路径转换成模型可理解的输入。

## 收益与代价

### 收益

- 保留重新生成、回退和多方案探索的完整历史；
- 当前分支仍能投影成模型熟悉的线性消息数组；
- 模型、思考级别和工具状态天然具有分支语义；
- append-only 日志便于 JSONL 写入、审计和崩溃恢复；
- compaction 不破坏原始历史；
- Session 领域逻辑不依赖具体存储介质。

### 代价

- 比简单 `messages[]` 更难理解和实现；
- 必须维护 ID、parentId、leaf 和重放规则；
- compaction、branch summary 与完整历史之间需要清晰边界；
- JSONL 适合本地单进程，不适合多个进程并发写同一 Session；
- 存储实现必须维护日志、索引和派生状态的一致性。

如果产品只支持单线聊天，`messages[]` 就够用。只有需要回退、重新生成和保留多个方案时，会话树增加的实现成本才值得。

## 源码阅读顺序

理解设计后，可以按以下顺序对应源码：

1. [`harness/types.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/types.ts)：`SessionTreeEntry`、`SessionContext` 和 `SessionStorage`；
2. [`session/session.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/session/session.ts)：分支状态归约、上下文投影和 Session 领域 API；
3. [`session/memory-storage.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/session/memory-storage.ts)：使用 Array、Map 和 leafId 实现树操作；
4. [`session/jsonl-storage.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/session/jsonl-storage.ts)：将同一套语义持久化为 JSONL；
5. [`session/jsonl-repo.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/session/jsonl-repo.ts)：创建、打开、列出和 fork 多个 Session；
6. [`compaction/compaction.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/compaction/compaction.ts)：长会话压缩策略；
7. [`compaction/branch-summarization.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/compaction/branch-summarization.ts)：分支离开与摘要生成。

阅读时始终追踪三个问题：

```text
新 entry 如何连接到当前 leaf？
当前 leaf 如何决定活动分支？
活动分支如何投影成 SessionContext？
```

## 总结

Session 用会话树保存所有分支，用 Active Leaf 选择当前工作位置，再沿 root-to-leaf 路径恢复状态和模型消息：

```text
Session Tree
保存所有探索分支
        ↓
Active Leaf
选择当前工作位置
        ↓
Root-to-Leaf Path
形成当前分支
        ↓
状态重放 + 消息投影
        ↓
SessionContext
交给 AgentHarness 和模型
```

会话树只存在于应用和持久化层。真正请求模型时，Pi 仍然传入一组普通的线性消息，Agent Loop 和 Provider 都不需要知道背后有一棵树。
