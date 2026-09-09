---
title: Pi 源码解析（二）：Agent Loop 的生命周期与控制流
date: 2026-08-12 21:11:00
toc: true
recommend: true
categories:
- 大模型
- Pi
tags: 
- 大模型
- Pi
- 源码解析
cover: https://file-1305436646.cos.ap-nanjing.myqcloud.com/blog/banner/2026-08-12.webp
---

# Pi 源码解析（二）：Agent Loop 的生命周期与控制流

> Pi Agent Core 源码系列（2/8）｜上一篇：{% post_link 2026-08-11-Pi-架构概览 "Pi 架构概览" %}｜下一篇：{% post_link 2026-08-13-Pi-Agent "Agent：轻量状态封装" %}

[`agent-loop.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/agent-loop.ts) 处理一次 Agent Run：把消息交给模型，执行模型返回的工具调用，再把结果送回模型，直到没有后续工作。这篇先看循环和事件顺序，流式响应和工具执行细节留到第 4 篇。

## Agent Loop 是无状态执行内核

Agent Loop 接收上下文、运行配置、取消信号和模型流函数。它会在本次运行中追加消息，但不拥有跨运行状态，也不知道 Session、终端 UI 或磁盘存储。

{% mermaid %}
flowchart TD
    App[应用代码] --> Agent[Agent<br/>轻量状态封装]
    App --> Harness[AgentHarness<br/>会话级编排]
    Agent --> Loop[Agent Loop<br/>执行语义]
    Harness --> Loop
    Loop --> Models[pi-ai<br/>统一模型流]
    Loop --> Tools[AgentTool<br/>工具实现]
{% endmermaid %}

`Agent` 和 `AgentHarness` 分别调用 Agent Loop，它们不是逐层嵌套关系。前者把事件归约为内存状态，后者把事件接入 Session、资源快照和持久化边界。

## AgentEventSink 保证事件顺序

Agent Loop 不直接更新 UI 或写入 Session，而是把所有运行变化发送给一个可注入的函数：

```ts
export type AgentEventSink = (event: AgentEvent) => Promise<void> | void;
```

循环中的每次发送都会被等待：

```ts
await emit({ type: "message_end", message });
await emit({ type: "tool_execution_start", ...toolCall });
```

这个 `await` 使 Sink 成为顺序屏障。上层可以先保存请求工具的 assistant 消息，再展示工具开始状态；如果持久化或监听器尚未完成，循环不会越过这个事件。

它和 `EventEmitter` 的区别不在于都能“发事件”，而在于控制权：

| 对比项 | `AgentEventSink` | `EventEmitter` |
|---|---|---|
| 接收端 | 调用方注入一个函数 | 对象管理多个 Listener |
| 异步语义 | 循环等待返回的 Promise | 默认不等待异步 Listener |
| 顺序影响 | 事件处理属于运行过程 | 异步工作通常独立继续 |
| 主要用途 | 定义 Runtime 输出边界 | 通用订阅与广播 |

多订阅者由上层 `Agent` 或 `AgentHarness` 管理。Agent Loop 只依赖一个 Sink，因此无需知道事件最终用于状态、渲染、日志还是存储。

## 四层生命周期

Pi 把一次运行拆成四层：

```text
Agent Run
└── Turn
    ├── Message
    └── Tool Execution
```

一次 `prompt()` 或 `continue()` 对应一个 Agent Run，从 `agent_start` 开始，以 `agent_end` 结束。一个 Run 可以包含多个 Turn。

一个 Turn 包含一次 assistant 响应以及该响应触发的全部工具执行和 `toolResult` 消息。模型读取工具结果后再次响应时，才会开始下一个 Turn。

消息分为 `user`、`assistant` 和 `toolResult`。三者都有 `message_start` 与 `message_end`，只有流式 assistant 消息产生 `message_update`。

一次工具调用拥有独立的 `tool_execution_start`、可选的 `tool_execution_update` 和 `tool_execution_end`。最终结果还会转换成一条 `toolResult` 消息进入对话历史。

## 十种运行事件

| 层级 | 事件 | 含义 |
|---|---|---|
| Agent | `agent_start` | 本次运行开始 |
| Agent | `agent_end` | 底层循环不再产生新事件 |
| Turn | `turn_start` | 新一次模型交互开始 |
| Turn | `turn_end` | assistant 响应及其工具批次完成 |
| Message | `message_start` | 一条消息开始 |
| Message | `message_update` | assistant 流式快照更新 |
| Message | `message_end` | 一条消息完成 |
| Tool | `tool_execution_start` | 工具调用开始处理 |
| Tool | `tool_execution_update` | 工具报告中间进度 |
| Tool | `tool_execution_end` | 工具调用完成或失败 |

没有工具调用时，完整顺序如下：

```text
agent_start
└── turn_start
    ├── message_start/end     user
    ├── message_start         assistant
    ├── message_update...     text/thinking
    ├── message_end           assistant
    └── turn_end
agent_end
```

存在工具调用时，模型需要在下一 Turn 消化工具结果：

```text
agent_start
├── turn_start                Turn 1
│   ├── message_start/end     user
│   ├── message_*             assistant(toolCall)
│   ├── tool_execution_*      工具执行
│   ├── message_start/end     toolResult
│   └── turn_end
├── turn_start                Turn 2
│   ├── message_*             assistant(final answer)
│   └── turn_end
└── agent_end
```

`tool_execution_end` 与 `message_end(toolResult)` 描述不同事实。前者表示执行状态已经结束，后者表示结果已经成为对话历史的一部分。

## 双层循环如何决定是否继续

`runLoop()` 使用内外两层循环：

```ts
while (true) {
  while (hasMoreToolCalls || pendingMessages.length > 0) {
    // 注入等待中的 steering
    // 请求模型并执行本 Turn 的工具
    // 发出 turn_end
  }

  // Agent 原本准备停止时检查 follow-up
  // 没有 follow-up 才真正结束
}
```

内层循环处理工具调用链和 steering。工具结果要求模型继续推理；steering 则代表用户希望调整正在进行的任务，两者都会在当前 Turn 完整结束后进入下一 Turn。

外层循环只负责 follow-up。它在工具链和 steering 都耗尽、Agent 原本准备结束时检查队列，从而表达“完成当前任务后再做一件事”。

真正决定工具链是否继续的是下面这段代码。`toolResult` 先按原顺序写回上下文，随后才发出 `turn_end`：

```ts
// 一个 assistant 消息中可能包含零个或多个工具调用块。
const toolCalls = message.content.filter((c) => c.type === "toolCall");

const toolResults: ToolResultMessage[] = [];
hasMoreToolCalls = false;
if (toolCalls.length > 0) {
  // stopReason 为 length 表示输出被 token 上限截断，工具参数可能只有一部分。
  // 即使残缺 JSON 恰好可以解析，也不能安全执行，因此统一生成失败结果，让模型重试。
  const executedToolBatch =
    message.stopReason === "length"
      ? await failToolCallsFromTruncatedMessage(toolCalls, emit)
      : await executeToolCalls(currentContext, message, config, signal, emit);
  toolResults.push(...executedToolBatch.messages);

  // 只有整批工具都要求 terminate 时才停止自动发起下一轮 LLM 调用。
  hasMoreToolCalls = !executedToolBatch.terminate;

  // 工具结果按 assistant 中 toolCall 的源顺序进入对话历史。
  for (const result of toolResults) {
    currentContext.messages.push(result);
    newMessages.push(result);
  }
}

// 一个 Turn 包含一次 assistant 响应及该响应触发的全部工具结果。
await emit({ type: "turn_end", message, toolResults });
```

这段节选来自 [`agent-loop.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/agent-loop.ts)。它把“模型声明了工具调用”和“下一轮是否还需要模型”分成两个判断：前者决定是否执行，后者由整个工具批次的 `terminate` 结果决定。

控制优先级因此是：

```text
当前工具批次
→ turn_end
→ steering
→ 工具链结束
→ follow-up
→ agent_end
```

Steering 和 follow-up 都不会强行打断已经开始的 Provider 流或工具执行。它们被放在确定的 Turn 边界消费，避免上下文在半个响应中途改变。

## Turn 边界允许刷新运行快照

每个 Turn 完成后，Loop 会把 assistant 消息、工具结果、当前上下文和本次新增消息交给 `prepareNextTurn`。上层可以返回新的上下文、模型或 thinking level：

```text
Turn 1 使用快照 A
→ turn_end
→ prepareNextTurn()
→ Turn 2 使用快照 B
```

`AgentHarness` 利用这个边界提交 save point，并为下一 Turn 构建新的配置快照。Loop 只定义“何时可以换”，并不负责决定 Session 如何保存这些变化。

`shouldStopAfterTurn` 提供另一个安全出口。上层可以在完整 Turn 后优雅停止，例如先进行上下文压缩，再通过 `continue()` 恢复执行，而不需要取消正在运行的工具。

## 错误和取消仍然闭合事件序列

Provider 最终消息的 `stopReason` 为 `error` 或 `aborted` 时，Loop 不再执行其中可能残留的工具调用：

```text
message_end assistant(error/aborted)
→ turn_end
→ agent_end
```

工具自身抛错的语义不同。异常会被转换为 `isError: true` 的 `toolResult`，通常仍进入下一 Turn，让模型解释失败或选择其他工具。

`agent_end` 只表示底层循环不会再发事件。上层可能还在等待最后一个监听器、提交 Session 写入或清理运行状态；`Agent.waitForIdle()` 和 Harness 的 `settled` 才表示对应封装已经稳定。

## 源码阅读路径

阅读 [`agent-loop.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/agent-loop.ts) 时，先跟踪 `runAgentLoop()` 和 `runAgentLoopContinue()` 如何建立首个事件，再进入 `runLoop()` 查看双层循环。看到 `streamAssistantResponse()` 和 `executeToolCalls()` 时先把它们当作两个黑盒，第 4 篇再展开模型流与工具执行协议。

## 总结

Agent Loop 通过双层循环安排工具链、steering 和 follow-up 的先后顺序。`AgentEventSink` 每次都等待接收端完成，所以 UI、状态更新和 Session 写入都能按同一顺序处理。
