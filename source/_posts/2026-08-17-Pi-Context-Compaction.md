---
title: Pi 源码解析（七）：上下文压缩与分支记忆
date: 2026-08-17 21:11:00
toc: true
recommend: true
categories:
- 大模型
- Pi
tags: 
- 大模型
- Pi
- 源码解析
cover: https://file-1305436646.cos.ap-nanjing.myqcloud.com/blog/banner/2026-08-17.webp
---

# Pi 源码解析（七）：上下文压缩与分支记忆

> Pi Agent Core 源码系列（7/8）｜上一篇：{% post_link 2026-08-16-Pi-Agent-Session "Session 会话树与持久化" %}｜下一篇：{% post_link 2026-08-18-Pi-Resources "Skill 与 Prompt Template" %}

Coding Agent 在一次任务里会读取大量文件、执行多轮命令，并不断把工具输出追加到会话中。如果每轮都把完整历史发给模型，上下文窗口很快会被占满，请求成本也会继续增加。

Pi 把完整历史和模型当前使用的工作记忆分开保存，再用两种摘要方式限制工作记忆的长度：

- **Compaction**：当前分支过长时，把较早历史压缩成检查点摘要；
- **Branch Summary**：用户离开一个探索分支时，把该分支的经验带到目标位置。

压缩后，模型仍需要知道用户目标、已完成的工作、关键决定和下一步。因此 Pi 不会只按 token 数删除旧消息。

## 上下文长度也会影响信息质量

上下文窗口通常被描述为一个容量限制，但对 Coding Agent 来说，它还是一个记忆质量问题。

{% mermaid %}
flowchart LR
    U[用户目标] --> R[读取文件]
    R --> T[工具结果]
    T --> E[代码修改]
    E --> V[测试与验证]
    V --> N[新一轮推理]
    N --> R
{% endmermaid %}

随着循环继续，历史中会混合不同价值的信息：

| 内容 | 长期价值 |
|---|---|
| 用户目标、约束和偏好 | 高 |
| 已完成工作和关键决定 | 高 |
| 当前正在修改的文件与下一步 | 高 |
| 早期读取过的完整文件内容 | 中或低 |
| 已经消费过的命令输出 | 中或低 |
| 重复尝试和中间推理 | 通常较低 |

只按时间删除最旧消息，会把初始目标一起删掉；什么都保留，早期的文件内容和命令输出又会挤占空间。Pi 因此会总结较早历史，并保留近期原文。

## 完整历史保留在 Session 中

Session 继续保存完整记录，Compaction 只改变发给模型的上下文。

{% mermaid %}
flowchart TD
    Session[完整 Session Tree<br/>消息、工具结果、配置变化、分支记录]
    Session --> Audit[恢复、审计与树导航]
    Session --> Projection[上下文投影]
    Projection --> Summary[历史摘要]
    Projection --> Recent[近期原始消息]
    Summary --> Context[模型工作记忆]
    Recent --> Context
{% endmermaid %}

压缩后仍然可以：

- 查看和恢复原始会话；
- 在会话树中切换分支；
- 根据完整记录还原模型、Thinking Level 和 Active Tools 等状态；
- 重新生成或替换摘要；
- 审计摘要是否遗漏了重要事实。

Compaction 为模型生成新的工作摘要，不会改写原始历史。

## 两种摘要方式

Pi 面对的是两种不同的上下文增长。

{% mermaid %}
flowchart TD
    Growth[上下文增长]
    Growth --> Linear[当前分支持续变长]
    Growth --> Branch[用户探索并离开分支]
    Linear --> Compact[Compaction<br/>压缩较早历史]
    Branch --> BranchSummary[Branch Summary<br/>保留旁支经验]
{% endmermaid %}

### Compaction：纵向压缩时间历史

当前分支持续增长时，较早消息被总结，近期消息继续保留原文。

```text
较早历史 + 近期历史
        ↓
历史摘要 + 近期历史
```

### Branch Summary：横向压缩探索分支

用户从一条探索路径返回旧节点或切换到其他分支时，只总结离开分支相对于共同祖先的独有内容。

```text
共同历史 + 旧分支独有内容
        ↓ 切换分支
共同历史 + 旧分支摘要 + 新分支工作
```

两种机制都使用模型生成语义摘要，但它们的触发条件、选择范围和摘要用途不同。

## Compaction 流程

一次完整压缩可以分成五个阶段。

{% mermaid %}
flowchart LR
    Measure[1. 估算上下文] --> Partition[2. 选择安全边界]
    Partition --> Prepare[3. 准备摘要输入]
    Prepare --> Summarize[4. 模型生成摘要]
    Summarize --> Project[5. 重建工作记忆]
{% endmermaid %}

程序和模型的分工如下：

> 程序决定何时压缩、压缩哪一段以及保留哪一段；模型只负责把选中的历史总结成结构化记忆。

这种分工避免让模型同时决定边界和内容。边界属于一致性问题，应由确定性逻辑控制；摘要属于语义理解问题，更适合交给模型。

## 提前预留摘要空间

压缩不能等上下文完全占满后才开始，因为生成摘要本身也需要输入和输出空间。Pi 使用“上下文窗口减去预留空间”作为触发边界。

{% mermaid %}
flowchart LR
    subgraph Window[模型上下文窗口]
        Used[当前上下文]
        Reserve[摘要请求与输出预留]
    end
{% endmermaid %}

默认策略预留约 16K token。当当前上下文越过安全线时，就应该进入压缩流程，而不是继续冒险发起普通 Agent Turn。

核心模块只提供 Token 估算、阈值判断和压缩方法。`AgentHarness.compact()` 仍需要显式调用；是否在 Turn 之间自动调用，由上层应用决定。这里的安全线是压缩阈值，不代表存在一个后台压缩任务。

Token 估算采用混合策略：

1. 优先使用最近一次 Provider 返回的真实 usage；
2. 对该 usage 之后新增的消息做启发式估算；
3. 没有可靠 usage 时，才对全部消息使用字符近似。

这样既利用了 Provider 的真实统计，又能覆盖工具结果、用户消息等尚未经过下一次模型请求的尾部内容。

所谓“尾部内容”，是上一次 Provider 请求完成后在本地追加、但尚未包含进下一次模型请求的消息。例如模型返回 Tool Call 和 usage 后，Agent 执行工具产生 Tool Result，用户也可能追加 Steering；这些消息还没有被 Provider 计数，只能暂按文本约 4 字符一个 token、图片固定成本来估算。下一次模型请求完成后，新的 input usage 会覆盖它们，重新成为更准确的基线。

Pi 没有在 Agent 层追求精确 tokenizer：不同 Provider 和模型的分词方式不同，最终请求还包含协议包装、Tool Schema、图片和 Provider 专属字段。这里的目标只是判断是否进入危险区，配合预留空间即可吸收大部分估算误差。

## 保留近期原文

Pi 默认保留最近约 20K token 的原始上下文，较早内容才进入摘要。

{% mermaid %}
flowchart LR
    Old[较早历史<br/>生成摘要] --> Boundary[安全切分点]
    Boundary --> Recent[近期历史<br/>保留原文]
{% endmermaid %}

近期原文具有摘要无法完全替代的价值：

- 当前正在修改的代码和最新工具结果；
- 模型刚刚建立的局部推理状态；
- 尚未完成的操作细节；
- 最新错误信息和精确输出；
- 当前 Turn 中 tool call 与 tool result 的对应关系。

新的模型上下文由较早历史的摘要和近期原文共同组成。

## 切分点不能破坏对话结构

最简单的压缩方式是按 token 数量切一刀，但 Agent 历史不是普通文本。它包含 Turn、Tool Call 和 Tool Result 等结构关系。

例如，下面这个切分是危险的：

```text
摘要区域：assistant 调用 read_file
──────────── 切分点 ────────────
保留区域：toolResult 返回文件内容
```

模型会看到一个没有完整调用上下文的工具结果。反向切分同样会留下没有结果的 Tool Call。

Pi 用两层策略处理这个问题：

1. **协议安全**：先建立合法切分点，保留区不能从裸 Tool Result 或结构性 Entry 开始；如果 token 边界落在 Tool Result，就对齐到后面的合法 User、Assistant 或自定义消息，使一组 Tool Call 与 Tool Result 一起进入摘要区或保留区；
2. **语义连续**：合法切分点仍可能是同一个 Turn 中间的 Assistant。此时消息协议虽然有效，但保留内容缺少用户原始目标和前半轮进展，因此还需要 Turn Prefix Summary。

换句话说，合法切分点解决“消息能否发送给 Provider”，Turn Prefix Summary 解决“保留的后半轮能否被模型理解”。

{% mermaid %}
flowchart TD
    Candidate[候选切分点]
    Candidate --> User[User Message<br/>优先]
    Candidate --> Assistant[Assistant Message<br/>可能拆分 Turn]
    Candidate --> Custom[Custom Message / Branch Summary]
    Candidate -.不允许.-> ToolResult[裸 Tool Result]
    Candidate -.不允许.-> Structural[模型变化、标签、Leaf 等结构条目]
{% endmermaid %}

因此，Pi 会优先保证消息协议完整，再考虑能压缩多少内容。

`findCutPoint()` 从最新条目向前累计 token，只在预先筛出的合法位置落刀。如果落点不是 User Message，还会记录当前 Turn 的起点，供后续生成 Turn Prefix Summary：

```ts
export function findCutPoint(
  entries: SessionTreeEntry[],
  startIndex: number,
  endIndex: number,
  keepRecentTokens: number,
): CutPointResult {
  const cutPoints = findValidCutPoints(entries, startIndex, endIndex);

  if (cutPoints.length === 0) {
    // 没有合法消息切点时只能从边界起点保留。
    return { firstKeptEntryIndex: startIndex, turnStartIndex: -1, isSplitTurn: false };
  }

  let accumulatedTokens = 0;
  let cutIndex = cutPoints[0];

  // 从最新条目向旧历史累计，直到达到希望保留的近期 token 数。
  for (let i = endIndex - 1; i >= startIndex; i--) {
    const entry = entries[i];
    if (entry.type !== "message") continue;

    const messageTokens = estimateTokens(entry.message as AgentMessage);
    accumulatedTokens += messageTokens;
    if (accumulatedTokens >= keepRecentTokens) {
      for (let c = 0; c < cutPoints.length; c++) {
        if (cutPoints[c] >= i) {
          cutIndex = cutPoints[c];
          break;
        }
      }
      break;
    }
  }

  // 将紧邻切点的结构性条目纳入保留区，避免它们孤立在摘要边界之外。
  while (cutIndex > startIndex) {
    const prevEntry = entries[cutIndex - 1];
    if (prevEntry.type === "compaction") {
      break;
    }
    if (prevEntry.type === "message") {
      break;
    }
    cutIndex--;
  }

  const cutEntry = entries[cutIndex];
  const isUserMessage = cutEntry.type === "message" && cutEntry.message.role === "user";
  const turnStartIndex = isUserMessage ? -1 : findTurnStartIndex(entries, cutIndex, startIndex);

  return {
    firstKeptEntryIndex: cutIndex,
    turnStartIndex,
    isSplitTurn: !isUserMessage && turnStartIndex !== -1,
  };
}
```

完整实现位于 [`compaction.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/compaction/compaction.ts)。前置的 `findValidCutPoints()` 负责排除裸 Tool Result 和结构性 Entry，`findCutPoint()` 再负责满足近期 token 预算。

## 超大 Turn：近期窗口也可能装不下

有时单个 Turn 本身就非常大。例如用户要求分析整个仓库，模型连续读取大量文件并执行多轮工具调用。即使希望保留最近 20K token，切分点仍可能落在同一个 Turn 中间。

这里三个概念需要分开：

```text
Turn Start ───── Turn Prefix ───── 切分点 ───── Turn Suffix
                 生成摘要                       保留原文
```

- **切分点**是开始保留原文的位置；
- **Turn Prefix** 是当前 Turn 从 User 起点到切分点之前的消息；
- **Turn Prefix Summary** 是对这段前缀生成的桥接摘要，并不是另一个切分点。

例如 token 边界落在 Tool Result 时，Pi 可以把切分点移动到后面的 Assistant Tool Call，使新的 Tool Call 和结果都留在原文区。但这个 Assistant 仍位于当前 Turn 中间：协议已经安全，语义上却不知道“用户最初要做什么、前面读取了什么、为什么继续调用这个工具”。Pi 因此总结切分点之前的 Turn Prefix，再把摘要放到保留的 Turn Suffix 前。

Pi 不会强行把整个 Turn 都保留下来，而是把它分成两种记忆：

{% mermaid %}
flowchart LR
    History[更早历史] --> HistorySummary[普通历史摘要]
    Prefix[超大 Turn 前半部分] --> PrefixSummary[Turn Prefix 摘要]
    Suffix[超大 Turn 后半部分] --> Keep[保留原文]
    HistorySummary --> Final[最终 Compaction Summary]
    PrefixSummary --> Final
    Keep --> Context[新工作记忆]
    Final --> Context
{% endmermaid %}

Turn Prefix 摘要不需要重复描述整个项目，而是集中回答：

- 这一轮最初要完成什么；
- 前半段已经做了哪些工作；
- 理解保留后半段所必需的上下文是什么。

普通历史摘要承担项目级长期记忆；Turn Prefix Summary 只充当当前断点的连接器。两者与保留原文组合后，模型既能看到总体目标，也能理解为什么 Turn Suffix 从当前 Assistant 或工具调用继续。这种设计允许 Pi 在极端长 Turn 中继续前进，同时降低“保留了一半工作，却丢掉为什么这样做”的风险。

## 摘要是结构化检查点，不是自由发挥

Pi 要求模型按照固定结构生成摘要：

```text
Goal
Constraints & Preferences
Progress
  Done
  In Progress
  Blocked
Key Decisions
Next Steps
Critical Context
```

这套结构服务于后续 Agent，而不是面向人类写一篇漂亮总结。

它刻意强调：

- 当前目标和用户约束；
- 已完成、进行中和阻塞项；
- 关键决定及原因；
- 下一步顺序；
- 精确文件路径、函数名和错误信息。

固定格式减少了摘要风格随模型变化而漂移，也让后续模型更容易定位“当前做到哪里”。系统提示词还明确禁止摘要模型继续对话或回答历史中的问题，避免摘要请求变成另一次 Agent Turn。

## 增量摘要：避免反复总结完整历史

长会话可能多次触发 Compaction。如果每次都重新发送全部原始历史，压缩本身很快也会变得昂贵。

Pi 使用增量摘要：

{% mermaid %}
flowchart LR
    A[历史 A + B] --> S1[Summary A+B]
    S1 --> Update[旧摘要 + 新历史 C+D]
    C[新历史 C + D] --> Update
    Update --> S2[Summary A+B+C+D]
    E[近期历史 E] --> Context[工作记忆]
    S2 --> Context
{% endmermaid %}

第二次压缩只需要提供：

- 上一次摘要；
- 上次保留但现在已经变旧的新历史；
- 更新摘要的明确指令。

原始历史仍然保存在 Session 中，但不会重复进入摘要请求。这使压缩成本随“新增工作量”增长，而不是随“完整会话长度”无限增长。

增量摘要的代价是误差可能累积，因此 Pi 还会确定性地保留部分容易丢失的信息。

## 文件操作是独立记忆通道

Coding Agent 最容易在多轮摘要中丢失的信息之一，是精确文件路径和文件操作状态。仅依赖语言模型总结，可能把“读取过”和“修改过”混在一起，也可能遗漏早期文件。

Pi 会额外提取两类信息：

```text
Read Files
Modified Files
```

这些信息同时存在于：

- 摘要文本中，供后续模型直接阅读；
- 结构化 details 中，供下一轮 Compaction 继承。

{% mermaid %}
flowchart TD
    Messages[被压缩消息] --> LLM[语义摘要]
    Messages --> Extract[确定性文件操作提取]
    Previous[上一次 Compaction Details] --> Extract
    LLM --> Result[Compaction Result]
    Extract --> Result
{% endmermaid %}

这是一种混合记忆设计：模型负责高层语义，程序维护可以可靠计算的事实。

## Branch Summary：带着旁支经验返回

Compaction 处理当前分支的时间增长，Branch Summary 处理会话树中的横向探索。

假设会话树如下：

{% mermaid %}
flowchart TD
    A[共同历史] --> B[共同祖先]
    B --> C[旧分支：方案一]
    C --> D[修改文件]
    D --> E[测试失败<br/>当前 Leaf]
    B --> F[目标分支：方案二]
{% endmermaid %}

用户从 `E` 跳到 `F` 时，目标分支天然拥有 `A → B`，但看不到 `C → D → E`。Pi 会：

1. 找到旧 Leaf 与目标节点的最深公共祖先；
2. 只收集公共祖先之后、旧分支独有的条目；
3. 在模型预算内优先保留该分支最近的工作；
4. 生成一条明确标注为“旁支探索”的摘要；
5. 将摘要带到目标位置继续工作。

{% mermaid %}
flowchart LR
    Old[旧分支独有历史] --> Summary[Branch Summary]
    Common[共同历史] --> Target[目标位置]
    Summary --> Target
    Target --> Continue[新分支继续工作]
{% endmermaid %}

Branch Summary 不是把两条分支的原始消息合并在一起。它只传递经验：做过什么、哪些方案失败、修改了哪些文件、为什么决定返回，以及如果以后继续这项工作应该做什么。

## Compaction 与 Branch Summary 的产品差异

| 维度 | Compaction | Branch Summary |
|---|---|---|
| 产品触发 | 当前上下文接近容量上限，或用户主动压缩 | 用户切换会话树位置 |
| 处理方向 | 沿当前分支纵向压缩旧历史 | 横向总结离开分支的独有历史 |
| 内容边界 | Token 预算和安全 Turn 边界 | 两条路径的最深公共祖先 |
| 保留内容 | 一段近期原始消息 | 目标分支本来的完整历史 |
| 摘要定位 | 当前任务的工作记忆检查点 | 旁支探索的经验说明 |
| 原始记录 | 保留在 Session | 保留在旧分支 |

二者的共同点是：都不删除历史，都把摘要建模为新的 Session Entry，并都保留文件操作信息。

## 失败时不应该污染 Session

摘要由模型生成，因此可能遇到取消、Provider 错误、输出异常或自定义 Hook 失败。产品层需要保证：摘要失败不能让会话进入“旧历史已经隐藏，但新摘要没有生成”的中间状态。

Pi 将准备、生成和持久化分成不同阶段：

{% mermaid %}
flowchart LR
    Prepare[准备边界] --> Generate[生成摘要]
    Generate -->|成功| Persist[追加 Session Entry]
    Generate -->|失败或取消| Keep[保持原 Session 不变]
{% endmermaid %}

只有得到有效结果后，Harness 才会追加 Compaction 或 Branch Summary Entry。原始历史始终存在，因此失败可以被观察、重试或交给扩展处理。

Harness 还允许 Hook 在摘要前取消操作，或直接提供自定义摘要。这使应用能够针对特定领域使用更严格的摘要器，而不必修改 Session 与上下文投影机制。

## 这套设计的收益

### 长任务可以持续运行

模型工作记忆不再与完整 Session 长度线性绑定。早期历史被压缩后，新的工具调用和消息仍有空间进入上下文。

### 最近细节与长期目标同时保留

近期消息保持原文，目标、进度和关键决定进入结构化摘要。它比纯截断更能维持任务连续性。

### 会话仍然可恢复和审计

Compaction 不删除原始 Entry。摘要错误不会永久破坏历史，也不会妨碍会话树导航。

### 分支探索不再完全丢失

用户可以尝试另一条方案，再带着失败经验和文件状态返回，而不需要把旁支完整内容塞入新上下文。

## 这套设计的代价

### 摘要不可避免地有损

模型可能遗漏细节、错误归纳决定，或在多次增量摘要后产生信息漂移。结构化格式和文件操作提取只能降低风险，不能消除风险。

### 压缩本身需要一次模型调用

Compaction 会增加延迟和 token 成本。预留空间过大，会过早触发压缩；过小，则可能没有足够空间生成可靠摘要。

### 切分策略需要在完整性与效率之间权衡

保留更多近期原文能提高局部准确性，但减少后续可用空间；保留过少则会让模型过度依赖有损摘要。

### 跨模型摘要可能改变表达质量

摘要使用当前选择的模型。切换模型后，摘要风格、推理能力和信息保真度可能变化，因此 Session 保留原始历史仍然很重要。

## 几个压缩参数

| 参数 | 调大后的效果 | 调小后的效果 |
|---|---|---|
| Reserve Tokens | 更早压缩，摘要空间更安全 | 更晚压缩，但更容易逼近窗口上限 |
| Keep Recent Tokens | 保留更多精确细节 | 为后续工作留下更多空间 |
| Summary Output Limit | 摘要更完整但更昂贵 | 摘要更短但遗漏风险更高 |
| Branch Input Budget | 保留更多旁支历史 | 更强调旁支近期结论 |

调整这些参数会直接影响压缩时机、请求成本和保留的原始细节。

## 完整流程

{% mermaid %}
flowchart TD
    Session[完整 Session Tree]

    Session --> Active[当前活动分支]
    Session --> Navigate[分支切换]

    Active --> Measure[估算上下文]
    Measure -->|接近安全线| Cut[选择安全切分点]
    Cut --> Old[较早历史]
    Cut --> Recent[近期原文]
    Old --> CompactSummary[结构化 Compaction Summary]

    Navigate --> Unique[离开分支独有历史]
    Unique --> BranchSummary[结构化 Branch Summary]

    CompactSummary --> Context[模型工作记忆]
    Recent --> Context
    BranchSummary --> Context

    Session --> Durable[恢复、审计与再次投影]
{% endmermaid %}

## 源码位置

理解产品设计后，再回到源码会更容易建立对应关系：

- [`compaction.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/compaction/compaction.ts)：Token 估算、安全切分点、增量摘要和超大 Turn；
- [`branch-summarization.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/compaction/branch-summarization.ts)：公共祖先、离开分支选择和旁支摘要；
- [`utils.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/compaction/utils.ts)：对话序列化和文件操作提取；
- [`session.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/session/session.ts)：Compaction-aware 上下文投影；
- [`agent-harness.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/agent-harness.ts)：压缩、分支导航、Hook 与持久化编排。

## 总结

Pi 在压缩时同时保留摘要、近期原文和完整 Session 历史：

- Session 保存完整、可恢复的历史事实；
- 程序根据 Token 预算和协议完整性选择压缩边界；
- 模型把较早历史转换为结构化工作记忆；
- 近期消息继续保留原文；
- 增量摘要控制重复成本；
- 文件操作通过确定性通道跨越多轮压缩；
- Branch Summary 让旁支探索以经验而不是原始消息的形式回到目标分支。

完整历史可以继续增长，而每轮发给模型的内容仍然被控制在可用范围内。
