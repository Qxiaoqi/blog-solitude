---
title: Pi 源码解析（四）：流式响应与工具执行
date: 2026-08-14 21:11:00
toc: true
recommend: true
categories:
- 大模型
- Pi
tags: 
- 大模型
- Pi
- 源码解析
cover: https://file-1305436646.cos.ap-nanjing.myqcloud.com/blog/banner/2026-08-14.webp
---

# Pi 源码解析（四）：流式响应与工具执行

> Pi Agent Core 源码系列（4/8）｜上一篇：{% post_link 2026-08-13-Pi-Agent "Agent 的状态管理" %}｜下一篇：{% post_link 2026-08-15-Pi-AgentHarness "AgentHarness 的状态和会话" %}

Agent Loop 的每个 Turn 分两步：`streamAssistantResponse()` 把模型事件合成一条 assistant 消息，`executeToolCalls()` 再处理其中的工具调用。工具只会在整条模型响应结束后执行。

## 一次 Turn 的执行管线

{% mermaid %}
flowchart LR
    Context[AgentMessage 上下文] --> Transform[transformContext]
    Transform --> Convert[convertToLlm]
    Convert --> Provider[streamFunction]
    Provider --> Assistant[AssistantMessage]
    Assistant --> Calls[提取 toolCall]
    Calls --> Validate[参数与策略校验]
    Validate --> Execute[执行工具]
    Execute --> Results[toolResult 消息]
    Results --> Next[下一 Turn]
{% endmermaid %}

模型流和工具执行共享同一份 Turn 上下文，但职责不同。模型只生成结构化调用声明；Runtime 等整条 assistant 消息完成后，才决定调用是否存在、参数是否合法以及如何调度。

## 请求 Provider 前先转换消息

Agent 内部允许 Session 摘要、UI 通知等自定义消息，Provider 只认识标准 `user`、`assistant` 和 `toolResult`。请求模型前会经过两次转换：

```text
AgentMessage[]
→ transformContext()   裁剪、压缩或注入上下文
→ AgentMessage[]
→ convertToLlm()       过滤或转换自定义消息
→ Message[]
→ Provider
```

`transformContext` 仍工作在 Agent 消息层，`convertToLlm` 是进入 pi-ai 标准协议的最后边界。随后 Runtime 使用 system prompt、标准消息和本 Turn 工具快照构造 `Context`。

API Key 在每次请求前重新解析，以支持会过期的 OAuth Token。具体 Provider 被隐藏在注入的 `streamFunction` 后面，Agent Loop 只消费统一的 `AssistantMessageEventStream`。

## 异步事件流如何形成一条消息

`streamFunction()` 返回的不是已经完成的响应数组，而是可以逐项等待的异步流：

```ts
for await (const event of response) {
  // start、text_delta、thinking_delta、toolcall_delta、done...
}
```

一次响应会随时间产生 `start`、内容增量和 `done`。内容事件分成文本、thinking 和 tool call 三组：

```text
text_start       thinking_start       toolcall_start
text_delta       thinking_delta       toolcall_delta
text_end         thinking_end         toolcall_end
```

每个内容事件同时提供两个视角：`delta` 是本次新增内容，`partial` 是截至当前已经聚合出的完整 `AssistantMessage` 快照。终端可以直接追加 `delta`，状态层则适合用 `partial` 覆盖当前流式消息。

Pi 在收到 `start` 时向上下文插入一个占位 assistant 消息。后续事件原位替换最后一项，`done` 或 `error` 到达后再用 `response.result()` 的最终消息覆盖它：

```text
start
→ context.messages.push(partial)
→ message_start

content event
→ context.messages[last] = event.partial
→ message_update

done/error
→ context.messages[last] = finalMessage
→ message_end
```

历史最终只保留一条完整 assistant 消息。`message_update` 是运行时观察数据，不会把每个 delta 都保存成独立消息。

源码中的关键点是：上下文最后一项始终表示“当前最新的 assistant 快照”，而不是不断追加 delta：

```ts
for await (const event of response) {
  switch (event.type) {
    case "start":
      // start 建立上下文中的占位消息，后续 delta 都原位替换它。
      partialMessage = event.partial;
      context.messages.push(partialMessage);
      addedPartial = true;
      await emit({ type: "message_start", message: { ...partialMessage } });
      break;

    case "text_delta":
    case "thinking_delta":
    case "toolcall_delta":
      if (partialMessage) {
        // event.partial 是 Provider 聚合后的最新完整快照，而不是单独的 delta。
        partialMessage = event.partial;
        context.messages[context.messages.length - 1] = partialMessage;
        await emit({
          type: "message_update",
          assistantMessageEvent: event,
          message: { ...partialMessage },
        });
      }
      break;

    case "done":
    case "error": {
      const finalMessage = await response.result();
      context.messages[context.messages.length - 1] = finalMessage;
      await emit({ type: "message_end", message: finalMessage });
      return finalMessage;
    }
  }
}
```

这是 [`streamAssistantResponse()`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/agent-loop.ts) 的主路径节选。完整实现还处理 Provider 没有发送 `start`、异步流没有显式 `done` 等兼容情况。

## toolcall_end 不是执行信号

`toolcall_start/delta/end` 表示模型正在生成工具名称和参数。收到 `toolcall_end` 时，同一 assistant 消息仍可能继续生成第二个调用、文本或 thinking，也可能最终因 token 上限停止。

Runtime 会等待整个模型流结束：

```text
toolcall_end
→ 工具声明生成完成
→ done
→ assistant message_end
→ 提取全部 toolCall
→ executeToolCalls()
```

这条边界防止 Runtime 根据尚未完成的响应提前执行副作用。模型表达“想做什么”，工具执行器决定“能否做以及怎样做”。

## 工具调用先经过确定的预检顺序

每个调用在执行前经过以下管线：

```text
查找工具
→ prepareArguments
→ schema 校验
→ beforeToolCall
→ AbortSignal 检查
→ execute
→ afterToolCall
→ tool_execution_end
→ toolResult message
```

`prepareArguments` 位于 schema 校验之前，用于兼容旧版或非标准 Provider 生成的参数形状。转换后的参数必须通过工具 schema，`beforeToolCall` 才能根据调用、参数和上下文决定是否阻止执行。

工具不存在、参数非法、Hook 阻止或操作取消都不会直接击穿 Agent Run。这些情况会形成 `isError: true` 的标准工具结果，让模型在下一 Turn 解释错误或修正调用。

`afterToolCall` 在执行结束后加工结果，可以替换 `content`、`details`、`usage`、`terminate` 和 `isError`。这些是字段级覆盖，不会对嵌套对象做深合并；Hook 抛错时，原结果会被替换为对应的错误结果。

## 进度更新也受事件顺序约束

工具的 `execute()` 可以通过 `onUpdate` 报告中间结果。回调本身是同步入口，但 Event Sink 可能异步，因此 Runtime 收集所有更新 Promise，并在工具完成或失败后等待它们：

```text
按报告顺序调用 update Sink
→ 等待所有已发出 update 完成
→ tool_execution_end
```

各次异步 Sink 的内部工作可能重叠，但 `tool_execution_end` 不会越过尚未结算的更新。工具 Promise 结算后，迟到的 `onUpdate` 会被忽略，因此 UI 不会在工具已经结束后又收到过期进度。

## 串行和并行模式保留不同顺序

全局 `toolExecution` 可以选择 `parallel` 或 `sequential`。单个工具也能声明 `executionMode: "sequential"`；一个批次中只要存在这种工具，整个批次都会串行，避免它与其他调用交错产生副作用。

串行模式的所有阶段都保持源顺序：

```text
start A → validate A → execute A → end A → result A
→ start B → validate B → execute B → end B → result B
```

并行模式采用“顺序预检、并发执行、有序写回”：

```text
start A → preflight A → start B → preflight B

execute A || execute B

end B → end A             按实际完成时间
result A → result B       按模型声明顺序
```

真实完成顺序适合驱动 UI，稳定的 `toolResult` 顺序则保证对话历史和 Provider payload 与原始 tool call 对应。

串行或并行并不是调用者临时决定的。Runtime 会检查整批工具，只要其中一个工具声明必须串行，整个批次就切换到串行路径：

```ts
async function executeToolCalls(
  currentContext: AgentContext,
  assistantMessage: AssistantMessage,
  config: AgentLoopConfig,
  signal: AbortSignal | undefined,
  emit: AgentEventSink,
): Promise<ExecutedToolCallBatch> {
  // 再次从 assistant 内容提取工具调用，确保实际执行以消息源顺序为准。
  const toolCalls = assistantMessage.content.filter((c) => c.type === "toolCall");

  // 单个工具的串行要求会提升为整个批次的串行要求。
  const hasSequentialToolCall = toolCalls.some(
    (tc) => currentContext.tools?.find((t) => t.name === tc.name)?.executionMode === "sequential",
  );

  if (config.toolExecution === "sequential" || hasSequentialToolCall) {
    return executeToolCallsSequential(currentContext, assistantMessage, toolCalls, config, signal, emit);
  }
  return executeToolCallsParallel(currentContext, assistantMessage, toolCalls, config, signal, emit);
}
```

## terminate 是批次级决定

单个工具结果可以设置 `terminate: true`，表示结果无需模型继续处理。但一个并行批次中，如果其他工具仍需要模型读取，提前终止就会丢失它们的语义。

Pi 只有在批次非空且每个最终结果都明确设置 `terminate: true` 时才跳过自动下一 Turn。混合批次仍把全部结果交给模型。

## 被截断的工具声明不会执行

assistant 因输出 token 上限结束时，`stopReason` 为 `length`。流式 JSON 修复可能把残缺参数解析成合法对象，甚至通过 schema，但参数在语义上仍可能缺少路径、命令尾部或限制条件。

Runtime 不执行这条消息中的任何工具，而是为每个调用发出闭合的 start/end 事件和错误 `toolResult`，让模型重新生成完整参数。这条规则把安全边界放在响应完整性上，而不是只依赖 JSON 和 schema。

## 工具异常与 Provider 失败走不同路径

工具抛出的异常会转换成错误结果，Agent Loop 通常继续下一 Turn。Provider 的最终 `stopReason` 为 `error` 或 `aborted` 时，当前 assistant 消息后不再执行工具，Turn 和 Run 直接结束。

```text
工具异常
→ toolResult(isError: true)
→ 模型可以恢复

Provider error/aborted
→ turn_end
→ agent_end
```

区分两者能让可恢复的执行失败留在对话协议中，同时避免从不完整的模型响应触发副作用。

## 源码阅读路径

先阅读 `streamAssistantResponse()`，跟踪 AgentMessage 转换、partial 占位和 final message 落盘。随后按 `executeToolCalls()`、`prepareToolCall()`、`executePreparedToolCall()`、`finalizeExecutedToolCall()` 和 `createToolResultMessage()` 的顺序阅读，就能看到一条工具声明如何变成稳定的对话结果。

## 总结

Runtime 等 assistant 消息完整结束后才执行工具，并按固定顺序完成预检、调用和结果写回。partial 只用于运行中展示，最终消息和 `toolResult` 才进入会话历史。如果响应被截断、取消或返回错误，Runtime 也不会从不完整的工具声明中执行操作。
