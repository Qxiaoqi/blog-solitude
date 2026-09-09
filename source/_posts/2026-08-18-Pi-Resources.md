---
title: Pi 源码解析（八）：Skill 与 Prompt Template
date: 2026-08-18 21:11:00
toc: true
recommend: true
categories:
- 大模型
- Pi
tags: 
- 大模型
- Pi
- 源码解析
cover: https://file-1305436646.cos.ap-nanjing.myqcloud.com/blog/banner/2026-08-18.webp
---

# Pi 源码解析（八）：Skill 与 Prompt Template

> Pi Agent Core 源码系列（8/8）｜上一篇：{% post_link 2026-08-17-Pi-Context-Compaction "上下文压缩与分支记忆" %}

`AgentHarnessResources` 包含两种 Markdown 资源：Skill 提供可按需加载的工作方法，Prompt Template 提供带参数的可复用 prompt。二者都由宿主应用加载并注入 Harness，但进入模型上下文的方式不同。

Skill 的名称、描述和位置可以进入 system prompt，完整正文只在模型选择或应用显式调用后加载。Prompt Template 不参与模型自主发现，应用通过 `promptFromTemplate()` 展开参数并启动一次普通 Agent Run。

## Resources 由宿主传入

Harness 只保存已经解析完成的对象：

```ts
export interface AgentHarnessResources {
  promptTemplates?: PromptTemplate[];
  skills?: Skill[];
}
```

磁盘发现、来源优先级、同名覆盖和诊断展示都属于宿主应用。这个边界让 Agent Core 不需要猜测用户目录，也不会在构造 Harness 时执行隐式文件 IO。

Resources 与 Tools 也不是同一层。Resources 是给模型或用户使用的指令材料；Tools 是带 schema 和 `execute()` 的运行能力。Skill 中即使包含脚本，实际副作用仍然必须经过当前 Turn 启用的工具。

## Skill 如何进入一次运行

从文件落盘到任务执行，一项 Skill 会经过四个阶段：发现、登记、选择、执行。

{% mermaid %}
flowchart LR
    Disk[Skill 目录] --> Loader[loadSkills<br/>发现与解析]
    Loader --> Resources[Harness Resources]
    Resources --> Catalog[System Prompt 中的目录]
    Catalog --> Choice[模型选择 Skill]
    Choice --> Read[读取完整 SKILL.md]
    Read --> Tools[调用 Read / Bash / 专用工具]
    Tools --> Result[Tool Result]
    Result --> Model[模型继续处理]
{% endmermaid %}

Skill 链路涉及三个主要模块：

| 模块 | 负责的事情 |
|---|---|
| [`skills.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/skills.ts) | 遍历目录、解析元数据、返回 Skill 和诊断信息 |
| [`system-prompt.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/system-prompt.ts) | 把可见 Skill 格式化成简短目录 |
| [`agent-harness.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/agent-harness.ts) | 保存资源快照，处理显式 Skill 调用 |

`AgentHarness` 接收加载结果，它自己不扫描磁盘。`loadSkills()` 使用 `ExecutionEnv` 访问文件，因此相同加载逻辑可以工作在 Node.js 或应用提供的其他执行环境中。

## 一个 Skill 是一个目录

常见结构如下：

```text
.agents/skills/
└── release-check/
    ├── SKILL.md
    ├── references/
    │   └── release-policy.md
    └── scripts/
        └── collect-artifacts.mjs
```

`SKILL.md` 是入口。frontmatter 中的 `name` 和 `description` 用来识别能力并帮助模型判断触发时机，正文写执行步骤。`references/` 保存按需查阅的材料，`scripts/` 放可以重复运行的确定性操作。

加载器发现 `SKILL.md` 后，会把它所在的目录视为完整能力包，内部子目录不再继续识别成独立 Skill。这样相对路径有了稳定基准，复制整个目录也能带走全部依赖。

目录带来的约束很直接：入口、文档和脚本属于同一项能力。相比把零散提示词塞进 system prompt，这种组织方式更容易审查，也更方便安装和升级。

## Skill 放在哪，由宿主应用决定

Agent Core 没有写死一个全局搜索路径。调用方把目录交给 `loadSkills()`，加载器才会开始工作。

项目级安装通常放在：

```text
<project>/.agents/skills/<skill-name>/SKILL.md
```

这是工具链采用的目录约定。宿主应用仍需把 `.agents/skills` 加入资源来源；少了这一步，文件虽然存在，Harness 也收不到它。

一个完整产品通常会合并几类来源：

{% mermaid %}
flowchart LR
    User[用户级 Skills] --> Merge[宿主应用合并资源]
    Project[项目级 Skills] --> Merge
    Plugin[插件携带的 Skills] --> Merge
    Merge --> Load[loadSkills]
    Load --> Harness[AgentHarness Resources]
{% endmermaid %}

`skills.ts` 按固定顺序遍历目录，并遵守 `.gitignore`、`.ignore` 和 `.fdignore`。文件访问经过 `ExecutionEnv`，解析失败记入 diagnostics；其中一个目录损坏不会阻止其他 Skill 加载。

这些行为说明加载器承担的是资源发现职责。来源优先级和错误展示仍然留在应用层。

## 为什么 system prompt 里只放目录

假设项目安装了五十项 Skill，把五十份正文全部放进 system prompt 会浪费大量 token。多数任务只会用到其中一两项，剩余内容还可能干扰模型判断。

Pi 采用渐进式加载：

{% mermaid %}
sequenceDiagram
    participant A as 宿主应用
    participant H as AgentHarness
    participant M as 模型
    participant R as Read 工具

    A->>H: 注入已解析的 Skills
    H->>M: name + description + location
    M->>M: 根据当前任务选择
    M->>R: 读取目标 SKILL.md
    R-->>M: 返回完整指令
    M->>M: 按指令继续执行
{% endmermaid %}

常驻上下文只保留名称、描述和文件位置。具体步骤等到命中任务后再进入上下文。

`formatSkillsForSystemPrompt()` 正好体现了这种渐进式加载：它只输出元数据，不读取或拼接 Skill 正文。

```ts
export function formatSkillsForSystemPrompt(skills: Skill[]): string {
  // 过滤禁止模型主动发现的 Skills。
  const visibleSkills = skills.filter((skill) => !skill.disableModelInvocation);
  if (visibleSkills.length === 0) return "";

  const lines = [
    "The following skills provide specialized instructions for specific tasks.",
    "Read the full skill file when the task matches its description.",
    "",
    "<available_skills>",
  ];

  // 每个字段都做 XML 转义，防止元数据破坏提示词结构。
  for (const skill of visibleSkills) {
    lines.push("  <skill>");
    lines.push(`    <name>${escapeXml(skill.name)}</name>`);
    lines.push(`    <description>${escapeXml(skill.description)}</description>`);
    lines.push(`    <location>${escapeXml(skill.filePath)}</location>`);
    lines.push("  </skill>");
  }

  lines.push("</available_skills>");
  return lines.join("\n");
}
```

源码见 [`system-prompt.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/system-prompt.ts)。`disableModelInvocation` 只影响模型能否在目录中发现该 Skill，不影响应用显式调用。

`description` 因此承担了一部分路由工作。写得太宽，模型会在无关任务中调用；写得含糊，真正需要时又匹配不到。有效描述通常要说清任务类型、触发条件和边界。正文可以很长，描述必须准确。

## 模型发现和显式调用

Pi 提供两条入口，适合不同控制强度。

### 模型自行选择

模型先看到 Skill 目录，再根据用户任务读取匹配的 `SKILL.md`。代码审查、文档整理、测试分析一类语义明确的工作，很适合这条路径。

```text
用户任务
→ 模型匹配 description
→ Read SKILL.md
→ 执行其中的流程
```

### 应用明确指定

调用 `harness.skill(name)` 时，Harness 直接查找资源，把完整 Skill 内容连同本次附加要求组成 prompt，然后启动一次 Agent Run。

完整正文在这里才被包装进请求，并明确告诉模型相对路径应从哪里解析：

```ts
/**
 * 将完整 Skill 内容包装成模型可识别的调用 prompt，并可追加本次用户要求。
 * location 和引用根目录帮助模型正确解析 Skill 内的相对路径。
 */
export function formatSkillInvocation(skill: Skill, additionalInstructions?: string): string {
  const skillBlock =
    `<skill name="${skill.name}" location="${skill.filePath}">\n` +
    `References are relative to ${dirnameEnvPath(skill.filePath)}.\n\n` +
    `${skill.content}\n</skill>`;

  return additionalInstructions ? `${skillBlock}\n\n${additionalInstructions}` : skillBlock;
}
```

这与 System Prompt 中的目录形成两阶段关系：目录负责发现，`formatSkillInvocation()` 才负责把选中的完整内容送进当前请求。

如果 frontmatter 设置了 `disable-model-invocation: true`，这项 Skill 不会出现在模型可见目录中，应用仍然可以显式调用它。发布、部署等需要人为选择的流程，可以借此减少误触发。

| 调用方式 | 谁做选择 | 完整正文何时进入上下文 |
|---|---|---|
| 模型发现 | 模型根据 description 判断 | 模型读取文件时 |
| 显式调用 | 用户或宿主应用指定名称 | Agent Run 启动前 |

## Prompt Template 是参数化的普通请求

Prompt Template 是一个 Markdown 文件，文件名形成模板名称，正文形成 prompt 内容。可选 frontmatter 提供 `description`；没有 description 时，加载器使用正文第一条非空行生成简短说明。

```text
prompts/
└── review.md
```

```markdown
---
description: Review one source file
---

Review $1. Focus on ${@:2}.
```

`loadPromptTemplates()` 对目录只读取直接 `.md` 子文件，不递归扫描；也可以接收显式 Markdown 文件路径。单个文件读取或 frontmatter 解析失败时，问题进入 diagnostics，其他模板仍能继续加载。

调用 `harness.promptFromTemplate("review", ["src/app.ts", "error handling"])` 时，Harness 从当前 Turn 资源快照中查找模板，替换参数后把结果作为普通 prompt 执行。支持的占位符包括 `$1`、`$2`、`$@`、`$ARGUMENTS`、`${@:N}` 和 `${@:N:L}`。

```text
Template content + args
→ formatPromptTemplateInvocation()
→ 普通 user prompt
→ Agent Loop
```

Skill 和 Template 的区别不在文件格式，而在控制方式。Skill 可以通过 description 让模型自行选择，并能引用同目录下的脚本与资料；Template 只能由应用显式调用，适合稳定地复用一种提问结构。

## 三条路径进入不同上下文位置

资源不会作为一个整体塞进消息历史。Harness 根据调用方式选择三条路径：

| 输入 | 进入位置 | 进入时机 |
|---|---|---|
| 模型可见 Skill 清单 | System Prompt | 宿主调用格式化函数时，只写 name、description、location |
| 显式 Skill | User Message | `skill()` 启动 Run 前写入完整正文与附加要求 |
| Prompt Template | User Message | `promptFromTemplate()` 展开参数后 |

`formatSkillsForSystemPrompt()` 会过滤 `disableModelInvocation`，对 XML 字段转义，再生成 `<available_skills>` 清单。动态 system prompt 回调能读取本 Turn 的 resources、模型和 active tools，因此宿主可以决定清单放在何处以及是否附加其他资源说明。

显式 Skill 会由 `formatSkillInvocation()` 包装成带 `name`、`location` 和引用根目录的 `<skill>` 块。Template 则只做参数替换，不获得额外协议语义。两者最终都成为普通 user 消息，后续流式响应和工具执行仍遵循 Agent Loop 的统一生命周期。

## 目录里的脚本如何运行

把脚本放进 Skill 目录，只解决了分发和定位。真正执行仍然经过 Agent 的工具系统。

`SKILL.md` 通常需要写清脚本的适用条件、解释器、参数、输出格式，以及运行前需要读取的参考文件。模型读到这些说明后，使用 Bash 或专用工具发起调用。

{% mermaid %}
sequenceDiagram
    participant M as 模型
    participant S as SKILL.md
    participant T as Bash / 专用工具
    participant P as Skill 脚本
    participant L as Agent Loop

    M->>S: 读取执行说明
    S-->>M: 路径、参数与结果约定
    M->>T: 发起工具调用
    T->>P: 执行脚本
    P-->>T: stdout / stderr / 结构化数据
    T-->>L: Tool Result
    L-->>M: 下一 Turn 继续处理
{% endmermaid %}

这段流程里，Skill 没有获得一条绕过 Agent Loop 的执行通道。模型仍要生成 tool call，工具仍要校验参数，执行结果仍以 `toolResult` 回到上下文。

脚本适合处理重复且容易出错的步骤，例如收集构建产物或检查发布状态。如果操作的参数约束严格或涉及敏感权限，更适合做成专用 Agent Tool。Tool 定义 schema 和执行规则，Skill 说明什么时候调用它，以及各步如何衔接。

## Skill 不会扩大权限

`SKILL.md` 可以要求模型修改文件、访问网络或运行命令，但最终权限由运行环境控制。

{% mermaid %}
flowchart LR
    Skill[Skill 中的操作说明] --> Model[模型生成 Tool Call]
    Model --> Schema[参数校验]
    Schema --> Hook[策略 Hook]
    Hook --> Approval[审批规则]
    Approval --> Sandbox[执行环境与沙箱]
    Sandbox --> Result[Tool Result]
{% endmermaid %}

当前 Turn 启用了哪些工具、`beforeToolCall` 是否放行、文件系统允许访问哪里、命令是否需要审批，这些条件共同决定操作能否执行。

因此，外部 Skill 值得按代码依赖的标准审查。重点看 `SKILL.md` 中的命令、脚本内容、外部下载地址和凭据使用方式。`disable-model-invocation` 只影响模型能否自主选择该能力，它不承担权限控制。

## Skill、Prompt Template 和 Extension Command

三者都可能由 `/xxx` 一类入口触发，背后的运行方式差别很大。

| 机制 | 保存的内容 | 适合解决的问题 | 对应用状态的控制 |
|---|---|---|---|
| Skill | 指令、参考资料、脚本 | 可复用的专项工作流 | 通过工具间接操作 |
| Prompt Template | 带参数的 prompt 文本 | 重复使用同一种提问格式 | 无直接控制 |
| Extension Command | TypeScript handler | UI、权限、工具集和 Session 编排 | 可以直接修改 |

例如 `/review file.ts` 很适合展开成 Prompt Template。代码审查需要额外清单和检查脚本时，可以做成 Skill。Pi 示例里的 `/plan` 会切换工具、限制命令并维护规划状态，所以由 Extension Command 实现。

判断时看任务需要什么：一段可复用文本、一套带资源的工作方法，还是应用级状态控制。

## 运行中的资源更新何时生效

Harness 在 Turn 开始时使用资源快照。磁盘上的 Skill 或 Prompt Template 发生变化后，宿主应用需要重新加载并调用 `setResources()`；已经发出的模型请求和正在执行的工具继续使用本轮快照，新内容在后续安全边界生效。

{% mermaid %}
flowchart LR
    Disk[磁盘内容变化] --> Reload[宿主重新加载]
    Reload --> Live[Harness 最新资源]
    Current[当前 Turn 快照] --> Finish[完成当前响应与工具]
    Live --> Next[下一 Turn 快照]
    Finish --> Next
{% endmermaid %}

这个规则避免同一 Turn 前半段看到旧说明、后半段突然改用新版资源。模型、工具和 system prompt 保持同一份视图，更新则留给下一轮。

## 源码阅读顺序

先读 [`skills.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/skills.ts) 和 [`prompt-templates.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/prompt-templates.ts)，比较两类文件如何被发现、解析并形成 diagnostics。再看 [`system-prompt.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/system-prompt.ts) 的 Skill 目录格式，以及 [`agent-harness.ts`](https://github.com/earendil-works/pi/blob/9e7582aa03e54f410fa9688197a3b64514e93400/packages/agent/src/harness/agent-harness.ts) 中 `createTurnState()`、`skill()`、`promptFromTemplate()` 和 `setResources()` 如何共享同一快照边界。

## 总结

Resources 把磁盘上的指令材料加载到 Turn 中，不会绕过 Agent Loop 创建新的执行通道。Skill 先暴露名称和描述，需要时再读入全文；Prompt Template 展开参数后，作为普通 user message 执行。
