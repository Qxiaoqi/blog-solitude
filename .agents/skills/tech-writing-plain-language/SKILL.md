---
name: tech-writing-plain-language
description: Revise technical writing into plain, concrete language without empty jargon, abstract filler, inflated framing, teacherly commentary, raw code dumping, or inconsistent article structure. Use when editing Chinese or mixed Chinese-English blog posts, architecture notes, design docs, summaries, headings, or callout sentences that contain phrases such as “心智模型”, “切面”, “全景”, “统一抽象”, “沉淀”, “收敛”, “承载”, “编排”, “治理”, “能力边界”, “很重要的工程事实”, “很关键的判断”, “最复杂”, “最成熟”, “很工程化”, “最值得关注”, or when a draft overuses source code, raw function names, and mixed heading numbering instead of explaining the system clearly.
---

# Tech Writing Plain Language

Use this skill to make technical prose sound like a pragmatic engineer wrote it plainly. Remove empty style, keep the technical meaning.

## Workflow

1. Scan the target files for obvious black话 and abstract filler.
- Start with titles, section headers, lead paragraphs, callout lines, and conclusions.
- When working in files, use `rg -n` with a compact watchlist before reading full context.

2. Read the full sentence before editing.
- Do not blindly replace words globally.
- Decide whether the wording is:
  - a real technical term that should stay
  - a vague phrase that should be simplified
  - a sentence that needs to be rewritten from scratch

3. Rewrite toward concrete language.
- Prefer direct subjects and verbs: `Gateway 负责...` instead of `Gateway 承载...`.
- Prefer actual things over meta concepts: `统一接口` instead of `统一抽象`, `整体结构` instead of `架构全景`.
- Prefer plain transitions: `更实际的做法是`, `可以这样理解`, `这说明`.

4. Preserve technical meaning.
- Keep real technical terms such as `PluginRegistry`, `ChannelPlugin`, `schema`, `queue`, `sandbox`, `provider`, `session`, `tool loop`.
- Keep `抽象` only when the text is specifically discussing interface design, abstraction boundaries, or an abstraction layer in code. If `接口`, `这一层`, or `统一接口` says the same thing, use the plainer form.
- Remove jargon, not precision.

5. Clean up presentation.
- Prefer Mermaid diagrams for architecture, data flow, lifecycle, state changes, and request paths.
- Use code blocks only when the exact code shape matters to the explanation.
- Core code can be shown when it is the shortest way to explain the mechanism.
- If the real code is too long, too noisy, or too coupled to implementation details, convert it to pseudocode first.
- Replace long source excerpts with a short summary plus Mermaid or prose when possible.
- Mention source function names only when they help the reader trace implementation. Do not build the article around raw function names.
- Explain behavior first, then name the file or function if it is still useful.
- Keep heading numbering consistent inside one article.
- Prefer one numbering scheme throughout the article, for example `## 1.`, `### 1.1`, `### 1.2`.
- Do not mix `一、二、三` with `1. 2. 3.` in the same article unless there is a strong reason.

6. Do a final tone pass.
- Remove slide-deck language, management-speak, and empty intensifiers.
- Make headings straightforward.
- If a sentence still sounds like a presentation deck, rewrite the whole sentence instead of swapping one keyword.
- Remove teacherly setup lines such as `最该先抓住`, `只留一句话`, `我会建议先记住`, `很重要的工程事实`, `很关键的判断`.
- Remove ranking or praise unless it is necessary to the technical point: `最复杂`, `最成熟`, `很工程化`, `非常关键`.

## Editing Heuristics

- Rewrite the sentence, not just the keyword, when the sentence is built around a vague concept.
- Prefer words a working engineer would actually say aloud.
- Treat titles and headings more aggressively than body paragraphs. Short, direct headings usually read better.
- Avoid repeated meta framing such as `最重要的心智模型`, `核心诉求`, `能力边界`, `形成闭环`, `统一收敛`.
- Avoid lecture-like framing that talks down to the reader or oversells the point.
- Avoid ranking modules or features unless the ranking itself matters to the explanation.
- Prefer diagrams and prose over large code dumps when introducing a system.
- Keep code excerpts focused on the core path. Trim setup, logging, and defensive noise when they are not the point.
- Use pseudocode when that preserves the logic better than copying a long real implementation.
- Avoid using source function names as section titles or as the main narrative device.
- Keep article formatting regular. Readers should not have to adapt to a new heading style halfway through.
- Do not overcorrect established domain terms just because they sound abstract.

## Fast Scan

Use this as a starting watchlist and adjust to the draft:

```bash
rg -n "心智模型|切面|全景|统一抽象|沉淀|收敛|承载|编排|治理|能力边界|核心诉求|语义化|落在|赋能|很重要的工程事实|很关键的判断|最该先抓住|只留一句话|我会建议先记住|最复杂|最成熟|很工程化|最值得关注|非常关键" <files...>
```

## Use Reference

Read [references/plain-language-guide.md](references/plain-language-guide.md) when you need a replacement table, examples, or a keep-vs-rewrite checklist for borderline terms.

Keep the original structure unless the user asks for a heavier rewrite.
