# Secretary Agent 迁移 OpenClaw 对照与实施方案

> 状态：建议方案
>
> 更新时间：2026-07-21
>
> 目标版本：OpenClaw 2026.5.29 及以上（飞书官方插件的最低版本）
>
> 核心原则：优先采用 OpenClaw 原生能力，只迁移业务规则和必要数据，不复刻已有基础设施代码。

## 1. 结论

Secretary Agent 可以迁移到 OpenClaw，并显著减少本项目自己维护的代码。推荐使用 OpenClaw 默认 `main` Agent 和唯一逻辑会话 `agent:main:main`，不要为了“行情 Agent”额外创建第二个 Agent，除非未来确实需要权限或数据隔离。

默认方案不开发事件插件，而是将当前 `events` 的用途拆到 OpenClaw 已有能力中：

- 稳定事实、持仓、交易规则：`MEMORY.md`
- 当日观察、普通事件流水：`memory/YYYY-MM-DD.md`
- 精确时间提醒、周期任务：Cron
- 从对话推断出的自然跟进：Commitments + Heartbeat
- 后台研究和定时任务运行状态：Background Tasks / Task Flow / Cron run history
- 对话轨迹：OpenClaw Session transcript
- 长期记忆整理：Memory Flush + Dreaming

只有满足以下任一条件时，才保留结构化 `events.sqlite` 并开发一个很薄的工具插件：

- 必须可靠查询“全部未闭环事项”，不能接受 Markdown 清单；
- 必须保存外部事件幂等键、原始消息 ID 或完整元数据；
- 必须按状态、时间、来源执行 SQL 统计；
- 必须满足审计、对账或可证明的事件回放要求；
- Webhook 事件量明显超过会话和每日 Markdown 适合承载的规模。

## 2. 迁移目标与非目标

### 2.1 目标

1. Telegram、飞书/Lark、Webhook 和 Cron 汇入同一个逻辑会话。
2. 保留持仓、偏好、关注项、交易规则和未闭环事项的连续性。
3. 使用 OpenClaw 自带的会话、压缩、长期记忆、调度、渠道、后台任务和审计能力。
4. 删除重复基础设施，减少 Python 常驻服务和自定义配置。
5. 适配当前 2 vCPU、约 2 GB 内存的主机，使用远程模型，不运行本地大模型和浏览器自动化。
6. 支持渐进迁移和快速回滚，避免渠道或定时任务重复发送。

### 2.2 非目标

- 不逐行移植 Pydantic AI 的消息序列化和工具循环。
- 不把现有 `secretary_v2.db` 强行写入 OpenClaw 内部数据库。
- 不迁移所有旧会话为 OpenClaw 活跃上下文；旧库保留为只读归档即可。
- 不在第一阶段引入多个 Agent、MCP、外部向量库、Docker 或复杂插件。
- 不把每个行情 Tick 都写入唯一会话或触发一次模型调用。

## 3. 当前架构摘要

当前系统把状态分为三层：

1. `memory.md`：稳定偏好、持仓、计划、交易规则和长期跟踪项。
2. SQLite：
   - `events`：业务事件、提醒、回复和 open/resolved/promoted 状态；
   - `messages`：按 `session_key` 隔离的对话历史；
   - `scheduled_tasks`：持久化调度定义；
   - `agent_events`：Agent 行为审计；
   - `subagent_runs`：后台研究状态。
3. 运行时动态上下文：每次运行注入 `memory.md`、open events、最近 events、当前时间和来源渠道。

实现依据：

- 会话键和历史装载：[runtime.py](../secretary_v2/runtime.py#L204)
- Memory 与事件动态上下文：[runtime.py](../secretary_v2/runtime.py#L651)
- 数据表定义：[memory.py](../secretary_v2/memory.py#L246)
- 调度任务入口：[scheduler.py](../secretary_v2/scheduler.py#L195)
- Webhook 事件记录：[http_channel.py](../secretary_v2/channels/http_channel.py#L47)

## 4. 目标架构

```text
Telegram DM ───────────────┐
Feishu/Lark DM ────────────┤
Control UI / TUI ──────────┤
Webhook mapping ───────────┼──> agent:main:main
Cron --session main ───────┤           │
Heartbeat / Commitments ───┘           ├── Session transcript
                                       ├── MEMORY.md
                                       ├── memory/YYYY-MM-DD.md
                                       ├── memory search / Dreaming
                                       ├── Cron history / Task ledger
                                       └── 可选 events.sqlite 插件
```

关键约束：

- `agent:main:main` 是唯一“逻辑会话键”，不是永不压缩的无限聊天记录。
- Session 负责当前推理连续性，Memory 负责跨压缩/重启的长期连续性。
- 原始行情、K 线、指标和大批量 Webhook 数据仍应存放在外部数据源或专用文件/数据库中。
- Webhook 应先完成验签、去重和必要的持久化，再决定是否唤醒 Agent。

## 5. 能力迁移对照

| 当前能力 | OpenClaw 原生能力 | 迁移决策 | 额外配置 |
|---|---|---|---|
| Pydantic AI Agent 循环 | Embedded Agent Runtime | 直接替换 | 无 |
| DeepSeek/Anthropic/OpenAI 配置 | Provider + model catalog | 直接替换 | 通过 onboarding 配置一个主模型 |
| Telegram 长轮询、重连、分片、outbox | Telegram Channel | 删除自维护实现 | 渠道登录和 owner allowlist |
| 飞书 WebSocket、重连、文件发送 | 官方 `@openclaw/feishu` 插件 | 删除自维护实现 | `openclaw channels login --channel feishu` |
| HTTP `/hooks` | Gateway Webhooks + mappings | 直接替换 | 一个固定 mapping |
| `messages` 表和 `session_key` | Session store + transcript | 删除自维护实现 | `dmScope=main` |
| 手工历史裁剪/压缩 | Auto-compaction + pruning | 删除自维护实现 | 默认即可 |
| 压缩前保存长期信息 | Memory Flush | 删除自维护实现 | 默认开启 |
| `memory.md` | `MEMORY.md` + daily memory | 内容迁移 | 文件改名并整理 |
| Memory 全文/语义检索 | `memory_search` / `memory_get` | 直接采用 | 默认 memory-core；可先只用 FTS |
| 周期性 memory consolidation | Dreaming | 删除自定义 Cron | `/dreaming on` |
| `events.status=open` | Commitments、Cron、`memory/open-items.md` | 原生优先 | Commitments 可选开启 |
| `events.status=logged` | Session transcript + daily memory | 不迁移为独立表 | 无 |
| `events.status=promoted` | Dreaming promotion | 直接采用 | Dreaming |
| `agent_events` | Logs、Audit、Background Tasks、Cron runs | 删除自维护表 | 无 |
| APScheduler + `scheduled_tasks` | OpenClaw Cron | 直接替换 | 逐项创建任务 |
| pending response 检查 | Commitments + Heartbeat | 优先替换 | Commitments + 简短 HEARTBEAT.md |
| 精确提醒 | Cron one-shot/recurring | 直接替换 | `--session main` |
| 静默跳过 `NO_ACTION` | `NO_REPLY` / `HEARTBEAT_OK` | 改用 OpenClaw 约定 | 提示词调整 |
| 后台 deep research | Sub-agents / Background Tasks | 直接替换 | 默认并发限制 1 |
| `subagent_runs` | Background Task ledger / Task Flow | 删除自维护表 | 无 |
| Skill loader | OpenClaw workspace skills | 迁移技能文本 | 将技能放入 workspace `skills/` |
| `market_calendar` Python 工具 | Skill + 窄脚本或 Longbridge CLI | 保留业务能力，不做运行时插件 | 一个 workspace skill |
| `send_message` | Message tool + channel delivery | 直接替换 | 渠道已配置即可 |
| `db_query` / `db_execute` | 默认不保留 | 先移除 | 仅可选事件插件需要 |
| `system_review` SQL | `openclaw tasks audit`、Cron runs、logs | 改为原生审计 | 可保留一个低频异常检查 |
| `manage.sh` | Gateway daemon / systemd user service | 删除 | onboarding 安装 daemon |

参考：

- [OpenClaw Main Session](https://docs.openclaw.ai/concepts/main-session)
- [OpenClaw Memory](https://docs.openclaw.ai/concepts/memory)
- [OpenClaw Cron CLI](https://docs.openclaw.ai/cli/cron)
- [OpenClaw Heartbeat](https://docs.openclaw.ai/heartbeat)
- [OpenClaw Commitments](https://docs.openclaw.ai/concepts/commitments)
- [OpenClaw Background Tasks](https://docs.openclaw.ai/automation/tasks)
- [OpenClaw Feishu/Lark](https://docs.openclaw.ai/channels/feishu)
- [OpenClaw Webhooks](https://docs.openclaw.ai/webhook)

## 6. 最小配置策略

### 6.1 保留默认 main Agent

为降低复杂度，第一阶段只使用 OpenClaw 默认 Agent：

```text
agentId: main
workspace: ~/.openclaw/workspace
sessionKey: agent:main:main
```

不要创建 `market`、`secretary`、`hooks` 等额外 Agent。未来只有出现以下情况时才拆分：

- 多用户之间必须隔离私人记忆；
- Webhook 输入不可信，需要严格限制工具；
- 行情任务与日常秘书使用不同模型、权限或工作区；
- 后台任务量导致 main session 长时间阻塞。

### 6.2 唯一逻辑会话

推荐显式设置：

```bash
openclaw config set session.dmScope main
openclaw config set session.reset.mode none
```

效果：

- Telegram、飞书和 Control UI 的私聊进入 `agent:main:main`；
- Cron 和 Webhook 可显式指向同一会话；
- 自动 compaction 控制上下文体积；
- Memory Flush 在压缩前保存值得长期保留的信息。

注意：

- `/new` 和 `/reset` 仍会滚动当前 transcript；长期连续性必须依赖 Memory，而不是只依赖聊天记录。
- 群聊/频道默认保持独立会话。若行情源来自群聊，应将有价值事件转发到固定 Webhook mapping 或写入 Memory，而不是强行让所有群聊共用私人 main transcript。
- 仅适合单用户私人 Agent；多用户共享 main 会话可能造成隐私泄露。

### 6.3 模型

当前主机内存较小，使用远程模型，不在本机运行 Ollama 模型。DeepSeek 可通过官方 provider 插件和 onboarding 配置：

```bash
openclaw onboard --auth-choice deepseek-api-key
```

新部署应使用 `deepseek/deepseek-v4-flash` 或 `deepseek/deepseek-v4-pro`，不要继续配置即将退役的 `deepseek-chat` / `deepseek-reasoner` 兼容名称。参考 [OpenClaw DeepSeek](https://docs.openclaw.ai/providers/deepseek)。

### 6.4 渠道

飞书使用官方安装向导；Telegram 当前不使用 `channels login`，只需配置 Bot token 和访问策略：

```bash
openclaw channels login --channel feishu
```

Telegram 保持最小配置，并将 token 放在环境变量或 SecretRef 中：

```json5
{
  channels: {
    telegram: {
      enabled: true,
      dmPolicy: "allowlist",
      allowFrom: ["<owner numeric user id>"],
      groups: { "*": { requireMention: true } }
    }
  }
}
```

默认账号可通过 `TELEGRAM_BOT_TOKEN` 提供 token，避免把凭证直接写进文档或工作区。

安全基线：

- Telegram 使用单一 owner 数字 ID allowlist；
- 飞书使用自己的 `open_id` allowlist；
- 群聊默认要求 mention；
- 不启用动态 Agent 创建；
- 飞书优先 WebSocket，不暴露新的公网端口。

### 6.5 Webhook

只定义一个固定行情入口，并固定到 main session。不要允许请求体自由选择 `sessionKey`。

```json5
{
  hooks: {
    enabled: true,
    token: "<独立的高强度 secret>",
    path: "/hooks",
    defaultSessionKey: "agent:main:main",
    allowRequestSessionKey: false,
    allowedAgentIds: ["main"],
    mappings: [
      {
        match: { path: "market" },
        action: "agent",
        agentId: "main",
        sessionKey: "agent:main:main",
        wakeMode: "now",
        name: "Market event",
        messageTemplate: "行情事件：{{message}}",
        deliver: true,
        channel: "last"
      }
    ]
  }
}
```

上线前按实际 payload 调整 `messageTemplate`。静态 mapping 的 session key 不需要开启调用者自选 session。Webhook 应继续放在 Cloudflare Tunnel 或可信反向代理后面，Gateway 本身保持 loopback 绑定。

若外部系统需要高频推送，推荐增加一个不调用模型的前置聚合器：

```text
行情源 -> 验签/去重/聚合 -> 有意义的状态变化 -> OpenClaw Webhook
```

### 6.6 Heartbeat 与 Commitments

开启 Commitments，用它处理“用户没有明确设定时间，但适合稍后追问”的自然跟进：

```bash
openclaw config set commitments.enabled true
openclaw config set commitments.maxPerDay 3
```

`HEARTBEAT.md` 保持很短，例如：

```markdown
# Heartbeat

- OpenClaw 自动处理到期 commitments；无需在此重复扫描。
- 检查 memory/open-items.md 中是否有超过预期时间未更新的项目。
- 只在需要用户行动、风险显著变化或任务失败时发消息。
- 不重复发送同一事项，不发送“系统正常”类消息。
```

Commitments 到期由 OpenClaw 自动并入 Heartbeat 流程。Heartbeat 适合低噪声关注循环；精确到某个时刻的提醒必须使用 Cron；无事项时回复 HEARTBEAT_OK。Commitments 保留创建它的渠道上下文，不应当作跨渠道业务事件总账；跨渠道未闭环事项仍放在 `memory/open-items.md` 或 Cron 中。

### 6.7 Dreaming

用 OpenClaw Memory Dreaming 替换当前 `memory_consolidation`：

```text
/dreaming on
```

默认即可，不需要立即修改评分、阶段和模型配置。Dreaming 会维护自己的周期任务，并将符合阈值的信息从短期记忆提升到 `MEMORY.md`。参考 [OpenClaw Dreaming](https://docs.openclaw.ai/concepts/dreaming)。

## 7. Memory 与 Events 的 native-first 迁移

### 7.1 Memory 文件布局

建议工作区：

```text
~/.openclaw/workspace/
├── AGENTS.md
├── USER.md
├── SOUL.md
├── TOOLS.md
├── HEARTBEAT.md
├── MEMORY.md
├── memory/
│   ├── YYYY-MM-DD.md
│   └── open-items.md
└── skills/
    ├── secretary-core/
    │   └── SKILL.md
    └── market-calendar/
        └── SKILL.md
```

迁移规则：

- 将当前 `memory.md` 改为 OpenClaw 规范的 `MEMORY.md`。
- 只保留当前有效的持仓、关注项、用户偏好、交易规则、风险边界和长期计划。
- 删除模板注释、历史聊天摘录、已关闭事项和可从旧数据库查询的流水。
- 当前未闭环 events 导出到 `memory/open-items.md`，不要全部塞入 `MEMORY.md`。
- 旧的 memory 备份保留在本项目归档目录，不作为 OpenClaw bootstrap 文件加载。

首版不显式配置 embedding provider；没有可用 embedding 凭证时，OpenClaw 会自动退化为 FTS 关键词检索，内置 CJK 索引能够覆盖中文精确词、标的代码和事件 ID。只有实际发现语义召回不足时，再增加一个远程 embedding provider；不要在当前 2 GB 主机下载本地 embedding 模型。参考 [Memory Search](https://docs.openclaw.ai/concepts/memory-search)。

`MEMORY.md` 建议结构：

```markdown
# Long-term memory

## User preferences

## Current positions

## Watchlist and hypotheses

## Trading and risk rules

## Long-running plans
```

`memory/open-items.md` 建议结构：

```markdown
# Open items

- [ ] `item-id` | due: YYYY-MM-DD | source: telegram/webhook/cron
  简短事项、最近进展和期望的闭环条件。
```

### 7.2 Events 类型映射

| 当前 event | 默认迁移目标 | 说明 |
|---|---|---|
| `note/logged` | 当日 memory 或 session transcript | 通常不需要结构化迁移 |
| `remind/open` 且有精确时间 | Cron | 完成后删除/禁用任务 |
| `check/open` 自然跟进 | Commitment | 由 Heartbeat 到期触发 |
| 长期未闭环事项 | `memory/open-items.md` | 适合少量人工可读清单 |
| `response` | main session transcript | 同一会话已有完整回复上下文 |
| `triggered` | Webhook turn + task/log | 原始大 payload 留在外部源 |
| `resolved` | 删除/勾选 open item，dismiss commitment | 不需要永久注入上下文 |
| `promoted` | Dreaming promotion | 由 memory-core 管理 |

### 7.3 何时升级为事件插件

先运行 native-first 方案。只有实际出现查询或审计不足时，再添加事件插件。

插件只提供最小工具面：

```text
event_record
event_query
event_resolve
```

插件要求：

- 使用独立数据库，如 `~/.openclaw/workspace/data/events.sqlite`；
- 不直接读写 OpenClaw 内部 SQLite；
- Webhook 写入使用 `source_message_id` 或外部事件 ID 去重；
- 默认仅返回 open items 和摘要，不把全部流水注入每次提示词；
- 写工具设为 optional，并显式加入 tool allowlist；
- 数据库迁移和备份由插件自己负责。

OpenClaw 支持通过 `api.registerTool(...)` 注册这类工具，参考 [Tool plugins](https://docs.openclaw.ai/plugins/tool-plugins)。

## 8. 现有定时任务迁移

| 当前任务 | OpenClaw 方案 | Session | 是否保留自定义提示词 |
|---|---|---|---|
| `morning_briefing` | Cron | `main` | 保留精简版 |
| `morning_trend_scan` | Cron | `main` | 保留行情/交易日规则 |
| `pending_response_check` | Commitments + Heartbeat | `main` | 删除原 SQL 提示词 |
| `review_reminder` | Cron | `main` | 保留精简版 |
| `stale_check` | Heartbeat；必要时每周 Cron | `main` | 改为读 open-items |
| `memory_consolidation` | Dreaming | memory-core 管理 | 删除 |
| `system_review` | Tasks audit + Cron run history | `main` 或 command | 大幅简化 |
| 用户运行时创建的提醒 | OpenClaw Cron | `current`/`main` | 原生自然语言创建 |

示例：

```bash
openclaw cron create "0 8 * * *" \
  "生成早间简报。读取当前长期记忆、open items 和相关行情；仅在有重要变化时发送，其他情况回复 NO_REPLY。" \
  --name "Morning briefing" \
  --agent main \
  --session main \
  --tz Asia/Shanghai
```

注意：

- 创建后使用 `openclaw cron show <job-id>` 核验 session 和投递目标；
- 使用 `openclaw cron run <job-id> --wait` 做首次测试；
- 使用 `openclaw cron runs --id <job-id>` 检查执行和投递历史；
- 为每个任务明确 `Asia/Shanghai` 时区，避免沿用主机默认时区；
- 主会话任务适合提醒和上下文相关检查；重型独立报告才使用 isolated session；
- 确定性检查优先使用 command cron，避免无意义模型调用。

## 9. Skills 与业务规则迁移

### 9.1 `secretary-core`

不要把现有巨大系统提示词原样放进 `AGENTS.md`。拆分为：

- `AGENTS.md`：始终适用的短规则；
- `USER.md`：用户身份、语言和沟通偏好；
- `MEMORY.md`：当前事实与长期状态；
- `HEARTBEAT.md`：主动关注清单；
- `skills/secretary-core/SKILL.md`：提醒、复盘、事件闭环等按需工作流。

`AGENTS.md` 只保留这些核心约束：

1. `MEMORY.md` 中明确的当前持仓和规则优先于旧会话。
2. 当前价格、交易日和市场状态必须查询工具，不能依赖记忆。
3. 精确提醒使用 Cron，自然跟进使用 Commitments。
4. 无需通知时使用 `NO_REPLY` 或 `HEARTBEAT_OK`，不发送完成占位消息。
5. 原始行情不写入 `MEMORY.md`；只记录决策、假设变化和需要跟踪的状态。

### 9.2 Market calendar

推荐保留当前确定性市场日历逻辑，但改为 workspace skill 调用窄脚本或 Longbridge CLI，而不是开发 OpenClaw runtime 插件。这样可以继续支持 CN/HK/US、缓存和 fallback，又不增加 Gateway 配置。

若 Longbridge 已能直接返回交易日，skill 可以只规定：

- 先调用 Longbridge；
- 失败时调用现有 `market_calendar.py` 包装脚本；
- 不允许用普通工作日推断交易日；
- 输出统一为市场、日期、开闭市和数据来源。

### 9.3 Deep research

用 OpenClaw Sub-agents 替换 `subagent_runs.py`：

- 主 Agent 负责拆分任务和最终判断；
- 子 Agent 使用隔离 session；
- 完成后自动回报 main session；
- OpenClaw Background Tasks 记录 queued/running/succeeded/failed；
- 当前 2 GB 主机将并发保持为 1；
- 使用远程模型，避免浏览器和本地大模型同时运行。

参考 [OpenClaw Sub-agents](https://docs.openclaw.ai/subagents)。

## 10. 建议删除与保留的代码

### 10.1 迁移稳定后可删除

- `channels/telegram_channel.py`
- `channels/feishu_channel.py`
- `channels/http_channel.py`
- `scheduler.py`
- 自定义消息历史与 compaction 实现
- `session_locks.py`
- `subagent_runs.py` 的任务状态机部分
- `manage.sh` 和渠道 watchdog
- `agent_events` 及相关系统审计提示词
- 自定义 skills loader

删除应在 OpenClaw 完成验收并观察稳定后进行，不应与首轮部署同时进行。

### 10.2 建议保留或转换

- `memory.md` 内容：整理后迁移为 `MEMORY.md`；
- `secretary-core` 规则：拆到 AGENTS/HEARTBEAT/skill；
- `market_calendar.py`：作为窄脚本保留；
- Longbridge、搜索和研究方法：迁移为 skills；
- `secretary_v2.db`：只读归档，直到确认不再需要历史查询；
- 测试中的关键业务场景：转换为迁移验收清单。

## 11. 分阶段迁移

### 阶段 0：冻结与备份

1. 备份 `memory.md`、`secretary_v2.db`、skills 和实际定时任务列表。
2. 导出所有 `status='open'` 的 events。
3. 记录 Telegram/飞书 owner ID、默认发送目标和 Cloudflare Tunnel 路由。
4. 记录当前有效 Cron、最近执行时间和预期投递时间。
5. 不修改或删除旧数据。

### 阶段 1：OpenClaw 基础安装

1. 使用官方安装器或 npm 安装，不从源码构建，不使用 Docker。
2. 使用 onboarding 配置远程模型。
3. 配置 `dmScope=main` 和 `reset.mode=none`。
4. 运行 `openclaw config validate`、`openclaw doctor` 和 `openclaw security audit`。
5. 暂时不接管现有渠道和 Cron。

### 阶段 2：Memory 与 Skills

1. 在 OpenClaw workspace 创建精简 `MEMORY.md`。
2. 将 open events 导出为 `memory/open-items.md`。
3. 拆分 `secretary-core` 到 AGENTS/HEARTBEAT/skill。
4. 添加 market-calendar skill。
5. 开启 Dreaming。
6. 验证 `openclaw memory status --deep` 和中文检索。

### 阶段 3：单渠道试运行

1. 先只接入一个渠道，推荐 Telegram。
2. 验证私聊 session key 是 `agent:main:main`。
3. 测试重启、连续对话、Memory 写入和检索。
4. 测试一次手动 compaction，确认重要状态仍可恢复。

### 阶段 4：Webhook 与 Cron

1. 增加固定 `/hooks/market` mapping。
2. 小流量验证 payload 模板、鉴权、去重和回复投递。
3. 逐个迁移 Cron，每迁移一个就禁用旧系统对应任务。
4. 检查 Cron run history 和失败通知。
5. 开启 Commitments，并观察误触发和重复提醒。

### 阶段 5：飞书与正式切换

1. 通过官方插件接入飞书 WebSocket。
2. 验证 Telegram 与飞书私聊共享 main session。
3. 手动停止旧 Secretary 服务或至少关闭旧渠道与调度器。
4. 保留旧数据库和日志为只读，不立即删除代码。
5. 连续观察至少一个完整交易周。

### 阶段 6：收敛

1. 删除确认由 OpenClaw 替代的配置和服务代码。
2. 若 Markdown open-items 不够用，再实现可选事件插件。
3. 为 OpenClaw workspace 建立私有 Git 或备份策略。
4. 更新本项目 README，明确旧版与 OpenClaw 版运行边界。

## 12. 验收清单

### 会话与渠道

- [ ] Telegram 私聊 session key 为 `agent:main:main`。
- [ ] 飞书私聊进入同一 session key。
- [ ] 在 Telegram 提到的上下文可从飞书继续追问。
- [ ] 群聊仍保持隔离，不泄露 `MEMORY.md` 私人信息。
- [ ] Gateway 重启后 session、memory 和 Cron 仍存在。

### Memory

- [ ] 当前持仓和交易规则存在于 `MEMORY.md`。
- [ ] 当日行情观察写入 daily memory，而不是无限扩大 `MEMORY.md`。
- [ ] `memory_search` 能检索中文历史条目。
- [ ] Compaction 前 Memory Flush 生效。
- [ ] Dreaming 不会把短期噪声大量提升为长期记忆。

### Webhook

- [ ] 使用独立 token，不能复用 Gateway token。
- [ ] 请求不能自由选择 Agent 或 session key。
- [ ] 相同外部事件不会重复触发通知。
- [ ] 高频无变化行情不会启动模型调用。
- [ ] 有意义事件进入 main session 并按预期投递。

### Cron 与主动跟进

- [ ] 所有任务时区为 `Asia/Shanghai`。
- [ ] Cron 明确使用 `--session main`。
- [ ] 无事项时使用 `NO_REPLY`，不发送完成占位消息。
- [ ] Commitments 不会与 Cron 对同一事项重复提醒。
- [ ] `openclaw cron runs` 能看到成功、失败和投递状态。
- [ ] `openclaw tasks audit` 无持续异常。

### 资源

- [ ] 常态内存没有持续增长。
- [ ] Swap 不持续高频换入换出。
- [ ] 不在本机运行大模型。
- [ ] 不启用 Chromium/Playwright，除非升级内存。
- [ ] Sub-agent 并发保持为 1。
- [ ] 磁盘保留至少 5 GB 可用空间。

## 13. 回滚方案

1. 不删除旧 `secretary_v2.db`、`memory.md` 和 `config.yaml`。
2. 切换期间确保同一渠道只有一个系统消费消息。
3. 每迁移一个 Cron 就记录旧任务 ID 与新 OpenClaw job ID。
4. 若 OpenClaw 出现严重问题：
   - 禁用 OpenClaw 对应渠道；
   - 暂停 OpenClaw Cron；
   - 手动重新启用旧 Secretary 服务和旧任务；
   - 将切换期新增的重要记忆人工合并回旧 `memory.md`。
5. 不在两个系统同时开启主动提醒，避免重复消息和重复交易判断。

## 14. 首版推荐配置范围

首版只配置这些项目：

1. 一个默认 main Agent；
2. 一个远程主模型；
3. `session.dmScope=main`；
4. `session.reset.mode=none`；
5. Telegram 和飞书 owner 私聊；
6. 一个固定行情 Webhook mapping；
7. 3～4 个确有固定时点要求的 Cron；
8. 一个短 `HEARTBEAT.md`；
9. Commitments；
10. memory-core 默认索引与 Dreaming；
11. secretary-core 和 market-calendar 两个 workspace skills。

首版明确不配置：

- 多 Agent 路由；
- 调用者自选 Webhook session；
- Docker Gateway 或 Agent sandbox；
- 浏览器自动化；
- Ollama、本地 embedding 或外部向量数据库；
- 自定义 compaction provider；
- 自定义 Memory provider；
- 事件插件；
- MCP；
- 动态飞书 Agent 创建；
- 大量 per-channel/per-job 模型覆盖。

该范围能够覆盖当前主要需求，同时把配置面和故障面控制在最小范围内。

## 15. 官方资料

- [安装与系统要求](https://docs.openclaw.ai/install)
- [Main Session](https://docs.openclaw.ai/concepts/main-session)
- [Session 管理](https://docs.openclaw.ai/sessions)
- [Memory](https://docs.openclaw.ai/concepts/memory)
- [Compaction](https://docs.openclaw.ai/compaction)
- [Dreaming](https://docs.openclaw.ai/concepts/dreaming)
- [Cron CLI](https://docs.openclaw.ai/cli/cron)
- [Heartbeat](https://docs.openclaw.ai/heartbeat)
- [Commitments](https://docs.openclaw.ai/concepts/commitments)
- [Background Tasks](https://docs.openclaw.ai/automation/tasks)
- [Sub-agents](https://docs.openclaw.ai/subagents)
- [Webhooks](https://docs.openclaw.ai/webhook)
- [Telegram](https://docs.openclaw.ai/channels/telegram)
- [Feishu/Lark](https://docs.openclaw.ai/channels/feishu)
- [DeepSeek](https://docs.openclaw.ai/providers/deepseek)
- [Tool plugins](https://docs.openclaw.ai/plugins/tool-plugins)
