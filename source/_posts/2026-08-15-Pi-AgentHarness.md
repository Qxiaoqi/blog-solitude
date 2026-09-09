---
title: Pi 源码解析（五）：AgentHarness 如何管理状态和会话
date: 2026-08-15 21:11:00
toc: true
recommend: true
categories:
- 大模型
- Pi
tags: 
- 大模型
- Pi
- 源码解析
cover: https://file-1305436646.cos.ap-nanjing.myqcloud.com/blog/banner/2026-08-15.webp
---

# Pi 源码解析（五）：AgentHarness 如何管理状态和会话

> Pi Agent Core 源码系列（5/8）｜上一篇：{% post_link 2026-08-14-Pi-Tool-Runtime "流式响应与工具执行" %}｜下一篇：{% post_link 2026-08-16-Pi-Agent-Session "Session 会话树与持久化" %}

`AgentHarness` 把 Session、模型配置、工具、队列、事件、Hook 和上下文压缩接到同一套运行流程中。它直接调用 Agent Loop，不重复实现模型和工具循环。

它主要处理三件事：每个 Turn 读到哪些状态，消息和配置变化按什么顺序写入 Session，以及错误或取消后何时才算真正结束。

## AgentHarness 的职责

Pi 的 Agent 体系可以分成三层：

{% mermaid %}
flowchart TB
    App[应用 / CLI / UI]

    subgraph Runtime[Agent 运行层]
        Agent[Agent<br/>轻量内存状态]
        Harness[AgentHarness<br/>会话级编排]
    end

    Loop[Agent Loop<br/>模型—工具循环]
    Models[pi-ai Models<br/>Provider 统一入口]
    Session[Session<br/>持久化状态树]
    Resources[Tools / Skills / Templates]

    App --> Agent
    App --> Harness
    Agent --> Loop
    Harness --> Loop
    Loop --> Models
    Harness <--> Session
    Harness <--> Resources
{% endmermaid %}

三层的分工如下：

| 层次 | 负责的事 |
|---|---|
| Agent Loop | 模型何时调用、工具如何执行、什么时候进入下一 Turn |
| Agent | 如何在内存中维护消息、流式状态、取消和基础队列 |
| AgentHarness | 如何让多次运行共享 Session、配置、资源、扩展和持久化语义 |

`Agent` 和 `AgentHarness` 都直接使用 Agent Loop，它们不是上下级嵌套关系。`Agent` 更适合轻量运行，`AgentHarness` 则面向完整应用。

## Run、Turn 和 Session

Run、Turn 和 Session 对应不同的时间范围。

{% mermaid %}
flowchart TB
    Session[Session：完整会话生命周期]
    Run1[Agent Run 1：一次 prompt 到结束]
    Run2[Agent Run 2：下一次 prompt 到结束]
    T11[Turn 1：一次模型响应 + 工具结果]
    T12[Turn 2：模型读取工具结果后继续]
    T21[Turn 1]

    Session --> Run1
    Session --> Run2
    Run1 --> T11
    Run1 --> T12
    Run2 --> T21
{% endmermaid %}

- **Session** 可以跨越很多次用户请求，并且能够恢复和分支。
- **Agent Run** 从一次 `prompt()` 开始，到 `agent_end` 结束。
- **Turn** 是一次 assistant 响应，以及这次响应触发的全部工具执行和结果。

一个 Harness 通常贯穿整个 Session：

```text
打开 Session
→ 创建 AgentHarness
→ prompt("问题一")
→ idle
→ prompt("问题二")
→ idle
→ compact 或切换分支
→ 继续 prompt
→ 关闭会话
```

因此，Harness 通常按会话隔离，而不是作为所有用户共享的全局单例。模型注册表、无状态工具定义和执行环境可以共享，但 Harness 自身包含会话级可变状态。

## 三类状态

Harness 同时维护实时配置、Turn 快照和 Session 持久化状态。三者的生效时间不同。

{% mermaid %}
flowchart LR
    Config[Harness 实时配置<br/>应用希望接下来怎么运行]
    Snapshot[Turn 快照<br/>当前这一轮承诺怎么运行]
    Session[Session 持久化状态<br/>已经发生了什么]

    Config -->|Turn 开始 / save point| Snapshot
    Snapshot -->|模型消息与工具结果| Session
    Config -->|安全边界提交配置变化| Session
    Session -->|重建上下文| Snapshot
{% endmermaid %}

### Harness 实时配置

实时配置代表“下一轮应该使用什么”，包括：

- 当前模型与 thinking level；
- 已注册工具和当前 active tools；
- system prompt 与资源；
- Provider 请求选项；
- steering、follow-up 和 next-turn 队列。

UI 或扩展可以在运行期间修改它。getter 会立即看到新值，但当前已经发出的请求不会被追溯修改。

### Turn 快照

Turn 快照代表当前这一轮已经冻结的运行条件，包括：

- 本轮模型；
- 本轮可见工具；
- 本轮 system prompt；
- 本轮消息上下文；
- 本轮资源与请求选项。

它是模型请求、工具校验和工具执行共同遵守的契约。

### Session 持久化状态

Session 记录已经提交的事实：

- user、assistant 和 toolResult 消息；
- 模型、thinking level 和 active tools 的变化；
- compaction、branch summary、label 和当前分支位置。

下一份 Turn 快照从 Session 和 Harness 最新配置重新构建，而不是继续依赖一份不断漂移的临时消息数组。

## 为什么当前 Turn 必须使用快照

假设 Turn 1 开始时的配置是：

```text
模型：Claude
工具：read、bash
```

请求发出后，用户切换到 GPT 并禁用 `bash`：

```text
Harness 最新配置：GPT + read
当前 Turn 快照：Claude + read + bash
```

Claude 已经看到 `bash` 的工具定义，因此它返回 `bash` tool call 是合法行为。如果工具执行阶段读取实时配置，就会出现自相矛盾：

{% mermaid %}
sequenceDiagram
    participant UI as 用户/UI
    participant H as Harness 实时配置
    participant P as Claude Provider
    participant E as 工具执行器

    H->>P: Claude + read + bash
    UI->>H: 切换 GPT，禁用 bash
    P-->>E: bash tool call
    E->>H: 查询最新工具列表
    H-->>E: 只有 read
    E-->>E: bash not found
{% endmermaid %}

模型按照请求时的工具定义做出了正确决定，执行阶段却用另一套规则否定了它。更危险的情况是同名工具在运行中更换了参数 schema 或实现：调用可能被新版 schema 拒绝，也可能以错误语义执行。

快照把一致性边界固定在 Turn：

{% mermaid %}
sequenceDiagram
    participant UI as 用户/UI
    participant S1 as Turn 1 快照
    participant P as Claude Provider
    participant E as 工具执行器
    participant S2 as Turn 2 快照

    S1->>P: Claude + read + bash
    UI-->>S2: 更新为 GPT + read
    P-->>E: bash tool call
    E->>S1: 使用本轮 bash 定义校验并执行
    E-->>S1: toolResult
    Note over S1: turn_end
    S1->>S2: 在边界切换快照
    Note over S2: 下一轮使用 GPT + read
{% endmermaid %}

规则如下：

```text
一个 Turn 内配置固定
Turn 之间允许配置变化
```

如果用户同时发送 steering 消息，它也会在当前 Turn 完整结束后，由下一份快照处理。当前 Provider 流和已经开始的工具不会被中途换模型或换规则。

> 快照通常是顶层结构复制，而不是任意对象的深拷贝。工具定义应当视为不可变值，通过 Harness setter 替换，不应在运行中原地修改现有工具对象。

`createTurnState()` 把 Session、Resources 和 Harness 实时配置汇合成一份 Turn 快照。下面的节选展示了快照从哪里取值：

```ts
private async createTurnState(): Promise<AgentHarnessTurnState<TSkill, TPromptTemplate, TTool>> {
  // Session.buildContext() 会应用会话树、压缩记录等规则，生成当前分支的模型上下文。
  const context = await this.session.buildContext();
  const resources = this.getResources();
  const sessionMetadata = await this.session.getMetadata();
  const tools = [...this.tools.values()];

  // 工具注册表和激活列表分离：只有 activeTools 会暴露给本轮模型。
  const activeTools = this.activeToolNames
    .map((name) => this.tools.get(name))
    .filter((tool): tool is TTool => tool !== undefined);

  let systemPrompt = "You are a helpful assistant.";
  if (typeof this.systemPrompt === "string") {
    systemPrompt = this.systemPrompt;
  } else if (this.systemPrompt) {
    // 动态 Provider 可以根据本轮模型、资源、工具和 Session 计算提示词。
    systemPrompt = await this.systemPrompt({
      env: this.env,
      session: this.session,
      model: this.model,
      thinkingLevel: this.thinkingLevel,
      activeTools,
      resources,
    });
  }

  return {
    messages: context.messages,
    resources,
    sessionId: sessionMetadata.id,
    systemPrompt,
    model: this.model,
    thinkingLevel: this.thinkingLevel,
    tools,
    activeTools,
  };
}
```

完整实现见 [`agent-harness.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/agent-harness.ts)。实际返回值还包含请求选项的浅复制；这里保留了决定上下文、模型、工具和资源一致性的字段。

## 一次 prompt 的处理过程

一次请求经历准备、运行、持久化、save point 和结算五个阶段。

{% mermaid %}
sequenceDiagram
    participant UI as 应用/UI
    participant H as AgentHarness
    participant S as Session
    participant L as Agent Loop
    participant M as pi-ai Models
    participant T as Tool

    UI->>H: prompt(text)
    H->>H: 获取 phase 锁
    H->>S: 构建当前分支上下文
    H->>H: 创建 Turn 快照
    H->>L: 启动 Agent Run

    L->>M: 使用快照请求模型
    M-->>L: assistant 流式事件
    L-->>H: message events
    H->>S: 持久化完成消息

    opt assistant 包含 tool call
        L->>T: 使用当前快照校验并执行
        T-->>L: toolResult
        L-->>H: toolResult message
        H->>S: 持久化 toolResult
    end

    L-->>H: turn_end
    H->>S: 提交 pending writes
    H->>H: save point
    H->>H: 创建下一 Turn 快照

    alt 仍有工具结果、steering 或 follow-up
        H->>L: 继续下一 Turn
    else 没有后续工作
        L-->>H: agent_end
        H-->>UI: settled / 返回最终消息
    end
{% endmermaid %}

请求中有三次明确的切换：

1. **启动边界**：读取 Session 和实时配置，生成本轮快照。
2. **Turn 边界**：当前模型和工具完成后，提交运行期间的变化。
3. **结算边界**：底层 Loop 已结束，待写数据和外部监听器也完成。

## Harness 从 Session 读取当前分支

Harness 不直接维护会话树。它在创建 Turn 快照时调用 `Session.buildContext()`，得到当前活动分支的消息、模型、thinking level 和 active tools；消息结束后再把新记录追加回 Session。会话树、活动 Leaf 和存储实现都由 Session 处理，下一篇 {% post_link 2026-08-16-Pi-Agent-Session "Session：会话树与持久化" %} 会单独展开。

## save point 保证写入顺序

模型消息必须按照因果顺序持久化：

```text
user
→ assistant(toolCall)
→ toolResult
```

但用户可能在 assistant 流式生成期间切换模型。如果立即把 `model_change` 写入 Session，历史可能变成：

```text
user
→ model_change(GPT)
→ assistant(Claude)
```

这错误地暗示 Claude 的消息是在切换 GPT 后生成的。

Harness 分两步处理运行中的配置变化：

{% mermaid %}
flowchart LR
    Change[运行中修改配置]
    Live[立即更新实时配置]
    Pending[进入 pending writes]
    Messages[当前 Turn 消息依次落盘]
    Save[save point]
    Persist[按 FIFO 提交配置变化]
    Next[下一 Turn 读取新状态]

    Change --> Live
    Change --> Pending
    Messages --> Save
    Pending --> Save
    Save --> Persist
    Persist --> Next
{% endmermaid %}

`save point` 表示：

> 当前 Turn 的正式消息，以及本轮期间已经接受的待写修改，都已经按照确定顺序提交。

这样写出的历史不会把配置变化插到错误位置。恢复时也只有两种状态：变化仍在 pending 队列，或者已经写入 Session。

`handleAgentEvent()` 把这个顺序写进了事件处理逻辑：

```ts
private async handleAgentEvent(event: AgentEvent, signal?: AbortSignal): Promise<void> {
  if (event.type === "message_end") {
    // 先保证消息持久化，再允许订阅者观察已提交状态。
    await this.session.appendMessage(event.message);
    await this.emitAny(event, signal);
    return;
  }

  if (event.type === "turn_end") {
    // 即使订阅者失败，也尝试冲刷 pending writes，避免已接受的修改丢失。
    let eventError: unknown;
    try {
      await this.emitAny(event, signal);
    } catch (error) {
      eventError = error;
    }

    const hadPendingMutations = this.pendingSessionWrites.length > 0;
    await this.flushPendingSessionWrites();
    if (eventError) throw eventError;

    // save_point 表示本轮消息和待写配置都已按顺序落盘。
    await this.emitOwn({ type: "save_point", hadPendingMutations });
    return;
  }

  await this.emitAny(event, signal);
}
```

这里省略了 `agent_end` 的结算分支。消息提交发生在广播 `message_end` 之前，pending writes 则等到 `turn_end` 订阅者执行后再统一落盘，两种写入因此不会交换顺序。

## 事件和 Hook：观察事实与参与决策

Harness 对外开放两类扩展能力，它们不应混为一谈。

{% mermaid %}
flowchart LR
    Lifecycle[Harness 生命周期]
    Events[事件订阅 subscribe]
    Hooks[决策 Hook on]
    UI[UI / 日志 / 指标]
    Policy[权限 / 策略 / 请求变换]

    Lifecycle --> Events
    Events --> UI
    Lifecycle --> Hooks
    Hooks --> Policy
    Policy -->|结果返回生命周期| Lifecycle
{% endmermaid %}

### 事件：观察已经发生的事实

事件适合 UI、日志、持久化观察和指标，例如：

- Agent、Turn 和消息开始或结束；
- 工具执行进度；
- 队列、模型、工具和资源变化；
- save point、abort 和 settled。

订阅者可以异步处理，Harness 会等待它们，以维持事件顺序。

### Hook：参与即将发生的决策

Hook 适合策略和扩展，例如：

- 在 Agent Run 前补充 system prompt；
- 在请求模型前调整上下文；
- 修改 Provider 请求选项或 payload；
- 阻止工具调用；
- 修改工具结果；
- 取消压缩或分支切换。

可以将二者简单区分为：

```text
subscribe：告诉我发生了什么
on：在发生之前让我参与决定
```

## 三种消息队列对应三种用户意图

Harness 没有把运行期间的新消息都塞进一个队列，而是区分处理时机。

| 队列 | 用户意图 | 消费时机 |
|---|---|---|
| steering | 调整当前任务的方向 | 当前 Turn 完整结束后，优先继续当前 Run |
| follow-up | 当前任务结束后再补充一项 | Agent 原本准备停止时 |
| next turn | 留给下一次主动请求 | 下一次 prompt/skill/template 开始时 |

{% mermaid %}
flowchart TD
    Current[当前 Turn 完整完成]
    Steering{有 steering?}
    FollowUp{有 follow-up?}
    End[agent_end]
    NextRun[下一次用户主动 prompt]
    NextTurn[消费 next-turn 消息]

    Current --> Steering
    Steering -->|有| Current
    Steering -->|无| FollowUp
    FollowUp -->|有| Current
    FollowUp -->|无| End
    End --> NextRun
    NextRun --> NextTurn
{% endmermaid %}

它们共同遵守一个原则：不会打断已经开始的 Provider 流或工具执行，而是在有明确定义的边界注入。

## 工具注册表与 active tools 为什么分开

工具存在两个不同概念：

```text
工具注册表
→ 应用当前拥有的全部工具实现

active tools
→ 当前 Turn 实际暴露给模型的工具子集
```

这种分离允许应用预先注册完整能力，再根据模式、权限或环境控制模型当前能看到什么。Session 只需要记录 active tool names；真正的执行函数仍由应用在恢复时提供。

运行期间修改工具集合时，规则和切换模型相同：

```text
当前 Turn 继续使用旧工具快照
→ turn_end / save point
→ 下一 Turn 使用新 active tools
```

## phase 控制哪些操作可以同时进行

Harness 将操作分成两类。

### 结构性操作

以下操作会读取或改变 Session 分支、上下文边界或运行快照，因此必须互斥：

- 发起新的 prompt、skill 或 template；
- 压缩上下文；
- 切换会话树分支。

它们只能从 `idle` 开始，避免两个操作同时改变 Session 的因果顺序。

### 运行中可以接受的操作

以下操作在运行期间有明确语义，因此可以被接受：

- steering、follow-up 和 next-turn；
- 修改未来 Turn 的模型、思考级别、工具和资源；
- abort。

Harness 没有追求“所有操作任意并发”，而是为每种变化规定一个可预测的生效边界。

## 压缩和分支切换只在 idle 时进行

Harness 负责保证压缩和分支切换只能从 `idle` 开始，并负责在摘要成功后提交 Session Entry；它不在普通 Turn 中自行改写上下文。Token 边界、摘要内容和活动分支的投影规则分别属于 Compaction 与 Session，详见本系列第 6、7 篇。

## 错误、取消和 settled

Harness 尽量让失败也形成正常可观察的生命周期。异常会被归一化为 assistant 失败消息，使 UI、Session 和恢复逻辑仍能看到闭合的消息与 Turn。

取消和优雅结束也需要区分：

```text
abort
→ 请求当前 Provider 和工具尽快停止

agent_end
→ Agent Loop 不再产生新事件

settled / waitForIdle
→ pending writes、监听器和 Harness 清理已经完成
```

应用如果要释放会话资源或开始结构性操作，应等待 settled 语义，而不是只看到 `agent_end` 就立即行动。

## 完整处理流程

{% mermaid %}
flowchart TB
    Input[用户输入 / 队列消息]
    Build[从 Session + 实时配置<br/>构建 Turn 快照]
    Provider[Provider 流式响应]
    Tools[使用同一快照执行工具]
    Persist[消息按因果顺序写入 Session]
    Save[save point<br/>提交 pending writes]
    More{还有工具结果或队列消息?}
    Refresh[读取最新配置<br/>创建下一 Turn 快照]
    End[agent_end → settled]

    Input --> Build
    Build --> Provider
    Provider --> Tools
    Tools --> Persist
    Persist --> Save
    Save --> More
    More -->|有| Refresh
    Refresh --> Provider
    More -->|无| End
{% endmermaid %}

前面这些功能都接在这条流程上：

- Session 提供可恢复的历史；
- Turn 快照保证单轮一致性；
- save point 保证消息与配置变化的提交顺序；
- 事件提供观察能力；
- Hook 提供受控修改能力；
- 队列把运行中的新意图放到安全边界；
- phase 阻止破坏因果顺序的并发操作。

## 几条运行规则

### 当前请求使用快照，未来请求读取实时配置

UI 和扩展可以在运行中修改设置，但当前模型请求和工具批次不会被中途换规则。

### Session 是已经发生事实的真相源

消息、配置变化、压缩和分支位置都以追加式条目表达，从而支持恢复和审计。

### 消息先提交，外部修改在 save point 提交

这保证 assistant/toolResult 链不会被运行期间的配置变化插入错误位置。

### 观察和决策分开

事件负责报告事实，Hook 负责在受控位置参与决策。

### 结构性操作必须互斥

结构性操作互斥，运行中变化则延迟到明确的 Turn 边界生效。

## 源码阅读顺序

理解设计后，再回到源码会更容易定位细节：

1. [`agent-harness.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/agent-harness.ts)：关注 Turn 快照、执行入口、事件处理和 save point；
2. [`types.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/types.ts)：理解事件、Hook 结果和 Session 条目；
3. [`session.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/session/session.ts)：理解状态树如何投影成模型上下文；
4. [`memory-storage.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/session/memory-storage.ts)：理解最小 Session 存储模型；
5. [`jsonl-storage.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/session/jsonl-storage.ts)：理解追加式持久化与恢复；
6. [`compaction.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/compaction/compaction.ts)：理解长会话的上下文边界。

## 总结

`AgentHarness` 把一次运行中的状态变化和 Session 写入排成固定顺序：

```text
Session 提供历史
→ Turn 快照冻结当前语义
→ Agent Loop 完成模型与工具循环
→ 消息按因果顺序持久化
→ save point 提交运行中变化
→ 下一 Turn 读取最新配置
```

事件、Hook、队列和工具都遵守这个顺序。当前 Turn 使用固定快照，运行中的修改在 save point 提交，下一个 Turn 再读取新配置。
