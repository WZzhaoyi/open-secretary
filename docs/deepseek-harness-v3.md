# Secretary Agent v3：DeepSeek Harness 配置基底与 Workspace 边界发现

> 状态：候选 v3 实现思路，进入 POC 前评审
>
> 更新时间：2026-08-21
>
> 目标平台：[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) developer preview
>
> 目标主机：2 vCPU、约 2 GB 内存；当前调查主机实际可见内存约 1.6 GiB

## 1. 执行结论

Secretary v3 的第一交付物不是 Workspace 插件，而是 **Secretary Distribution**：一套开箱可用、锁定版本的 DeepSeek Harness 配置基底，组合官方 Bundle、经过准入的第三方 Bundle、Profile 模板、权限策略和发布清单。

Workspace 是待验证的产品需求模型，不是预设的代码包。先用真实 Harness 和插件组合逐项复现 v2 行为，才能知道 Workspace 语义哪些已经由官方能力、配置或第三方插件满足，哪些只是适配问题，哪些才是必须自研的稳定缺口。

因此最终交付形态是条件式的：

1. **必选：Secretary Distribution**，负责开箱可用的运行组合和兼容性基线。
2. **可选：Secretary Workspace Extension**，只实现经过 POC 证明无法由现有组合满足的剩余语义；它可能是一个小插件、若干窄 adapter，也可能完全不存在。

如果存在一种插件组合能够完整复现项目需求，并通过资源、副作用和迁移测试，Secretary v3 可以只发布 Distribution，Workspace 自研代码量为零。不能为了体现“项目核心”而人为制造插件。

可行性结论分为两层：

| 判断 | 结论 | 说明 |
|---|---|---|
| 配置基底可行性 | **Go，优先实施** | 先组合 Harness 官方与第三方能力，形成可启动、可测试、可锁定的 Distribution |
| Workspace 自研边界 | **尚未确定** | 必须由 v2 契约回放和真实插件组合的缺口证据决定，不能由架构推演预先决定 |
| 2 vCPU / 约 2 GB 运行 | **Conditional Go** | 仅限远程模型、预构建部署、Agent 并发 1、受限工具输出、少量 live Session、无本地 Chromium/本地编码 Agent |
| 当前 1.6 GiB 主机 | **只允许受限 canary** | 机器已使用 swap，资源余量小；源码安装实验单进程峰值约 729 MiB，不能在该机安装或构建源码依赖 |
| 正式迁移 | **尚未 Go** | 尚无官方完整组合的 RSS 基线，必须通过最小运行时实测、24/72 小时 soak 和真实工作负载回放 |

v3 不再把“单会话”当成产品需求。真正需要验证的是：用户始终面对同一个持续演进的 Secretary 项目，同时每项工作拥有恰当的注意力边界。一个 Workspace 可以包含 Home、Work、Research 等多个 Session；共享身份和记忆，隔离原始 transcript，并按需检索或引用其他 Session。这个结果可以由纯配置和现有插件实现，也可以需要少量补充代码。

## 2. 需求基线与 Workspace 注意力假说

### 2.1 可观察需求

v2 的单实例、单会话是为了降低上下文管理复杂度，不是不可改变的业务语义。迁移 Harness 后，需求应重述为：

- 同一用户跨重启、跨渠道和跨任务仍被识别为同一个持续协作对象；
- Telegram、飞书、Webhook、CLI/Web 和 Cron 都能延续同一个 Secretary 项目；
- 身份、偏好、长期事实、承诺和任务状态保持一致；
- 无关任务的 transcript、tool result 和失败不会污染当前工作；
- 当前工作能获得完成任务所需的历史信息，而不是自动拼接全部历史；
- 跨任务信息能够被检索、引用和提炼，并保持来源与权限边界；
- 主动发送、Cron 和 Webhook 仍保持单实例 lease、持久幂等和可审计投递。

### 2.2 概念映射

| Secretary 概念 | Harness 映射 | 生命周期 |
|---|---|---|
| Secretary Project | 一个长期存在的产品实例 | 跨版本、跨进程 |
| Secretary Workspace | Harness Workspace + 所选配置/插件；必要时才增加自研状态 | 长期存在，绑定规范化工作目录和 owner |
| Home Session | 默认交互与分流会话 | 长期但可压缩、轮换 |
| Work Session | 一项承诺或任务的执行上下文 | 任务期间存活，完成后归档 |
| Research Session | 一次研究运行的上下文 | 单次运行，可恢复、归档 |
| Goal | 当前 Session 的明确目标 | 同 Session 持久化 |
| Run / Job | 一次后台执行或定时触发 | 有界、可取消、可重试 |
| Memory | Workspace 共享的提炼事实 | 跨 Session、可人工编辑、可迁移 |
| Transcript | Session 事件日志 | 不全局合并，不作为共享记忆本体 |

Harness 官方 Workspace 本身是持久目录记录和 Session 索引，并且**不会直接向模型注入提示词、工具或事件**。这是需要在 Distribution POC 中验证的边界：注意力装配可能已经能由官方 system-prompt、preset、query/reference 与现有 Memory 插件组合完成；只有仍存在稳定缺口时，才需要 Secretary 自研扩展。参考 [Workspace subsystem](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/workspace.md) 和 [Workspace package](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/workspace/workspace/README.md)。

### 2.3 待验证的注意力目标

~~~text
Secretary Workspace
├── Shared identity / policies
├── Memory revisions
├── Commitments and TaskIndex
├── Delivery / idempotency ledger
├── Home Session
│   └── 默认对话、收件和任务分流
├── Work Session: <task-id>
│   └── 单项任务的 transcript、goal、tools、compaction
└── Research Session: <run-id>
    └── 研究过程、引用、报告和恢复点
~~~

上图描述目标行为，不代表这些节点必须由自研插件实现。一次 turn 理想上由以下内容组装，而不是读取一个无限增长的全局 transcript：

~~~text
current session history
+ scoped system prompt
+ shared identity and policy
+ selected memory facts
+ current commitment / goal
+ bounded references to related sessions
+ permitted tools for this session preset
~~~

Harness 的 system-prompt 支持作用域 section 和动态上下文；per-session preset 可以让同一 Host 中不同 Session 使用不同工具与提示词；Session Query 和 Session Reference 提供受限的跨会话检索与引用。参考 [system-prompt](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/system-prompt/README.md)、[preset](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/preset/README.md)、[session-query](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/session-query/README.md) 和 [session-reference](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/context/session-reference/README.md)。

### 2.4 待验证的路由假说

以下是用于启动实验的最简假说，不是预先冻结的内部架构；应由真实通道插件行为和阶段 0 契约决定是否保留：

1. 外部 channel/chat/thread ID 是来源和回复目标，不直接等于 Workspace 或长期 Session。
2. 普通消息默认进入 Home Session。
3. 已绑定 taskId、researchRunId 或显式 thread 的消息进入对应 Work/Research Session。
4. Cron 触发时先定位 Workspace，再创建或恢复有明确 taskId 的 Session。
5. Session 完成后把结论、承诺变化和必要引用提升到 Workspace；原 transcript 留在 Session 中归档。
6. 只有一个实例持有渠道收件、主动发送和调度 lease；多 Session 不等于多发送者。

## 3. Distribution-first 的交付与条件式代码组织

### 3.1 Secretary Distribution

Secretary Distribution 是部署组合，不复制 DeepSeek Harness 源码。它负责：

- 锁定 Harness、Node、官方 Bundle 和第三方 Bundle 的精确版本与 integrity；
- 提供 development、test、production Profile/Patch 模板；
- 挂载通道、调度、研究、Memory 等已选 Bundle，以及经缺口评审批准的可选 Secretary 扩展；
- 给出默认模型、Session preset、工具白名单、并发和资源限制；
- 声明网络、文件、凭证、子进程和监听端口权限；
- 保存最终组合配置、release manifest、迁移与回滚说明；
- 由 CI 产出预构建 tgz 或不可变运行时制品。

Distribution 首先只描述组合和策略，不主动新增 Memory、Task、Delivery 等状态所有者，也不把真实凭证和运行数据提交到 Git。状态由哪个现有组件拥有，必须在组合验证中记录清楚。

### 3.2 Workspace Extension 是条件式产物

只有某项需求同时满足以下条件，才进入自研候选：

1. 已在锁定的 Harness 基底上验证官方能力和至少一个合理的第三方候选；
2. 无法通过 Profile/Patch、提示词、preset、权限或数据映射解决；
3. 第三方插件适配或上游贡献的成本、风险明显高于本地窄实现；
4. 缺口属于 Secretary 的稳定产品语义，而不是某个 Harness RC 的暂时 API 差异；
5. 已有可执行契约测试能够证明缺口和验收实现。

下表只是潜在缺口目录，不是开发 backlog：

| 模块 | 责任 |
|---|---|
| workspace-registry | 仅当现有通道和 Workspace 无法稳定映射 owner/channel/cwd |
| attention | 仅当 prompt/preset/query/reference/Memory 组合仍无法选择上下文 |
| memory | 仅当现有 Memory 方案无法满足人工编辑、备份和 revision 契约 |
| task-index | 仅当 Goal/Session/Jobs 无法表达 taskId、恢复和归档 |
| router | 仅当通道插件无法把消息、Cron 和 callback 路由到正确 Session |
| delivery | 仅当现有通道层无法提供幂等、重试、receipt 和单实例 lease |
| promotion | 仅当现有 compaction/Memory 无法把完成结果安全提升为共享事实 |
| migrations | 仅当确实新增了 Secretary 自有持久状态 |

若多个已证实缺口共享同一 Workspace 状态和生命周期，才将其聚合为 `dsh-secretary-workspace` Bundle。若只有一个窄缺口，应优先发布单一 adapter；若没有缺口，不创建任何 Secretary 运行时代码。

只有确认自研后，才冻结类似以下的最小 service 接口，避免提前设计一套无人使用的抽象：

~~~text
SecretaryWorkspace.resolve(origin) -> WorkspaceRef
Attention.assemble(workspaceId, sessionId) -> ContextSelection
TaskIndex.open/resume/archive(taskId) -> SessionRef
Memory.search/commit/revise -> MemoryRevision
Channel.deliver(message, idempotencyKey) -> DeliveryReceipt
Scheduler.dispatch(taskId, workspaceId) -> RunReceipt
~~~

### 3.3 仓库结构随证据演进

~~~text
secretary_v3/
├── distribution/
│   ├── profiles/
│   │   ├── development.patch.yml
│   │   ├── test.patch.yml
│   │   └── production.patch.yml
│   ├── manifests/
│   ├── permissions/
│   └── plugin-inventory.yml
├── evaluations/
│   ├── v2-contracts.md
│   ├── capability-matrix.md
│   ├── gap-analysis.md
│   └── decisions/
├── tests/
│   ├── contracts/
│   ├── lifecycle/
│   ├── fault-injection/
│   ├── replay/
│   └── resource/
├── packages/                       # 只有缺口 ADR 批准后才创建
│   └── workspace-or-adapters/
└── scripts/
    ├── compatibility-matrix.*
    ├── build-artifacts.*
    └── release-manifest.*
~~~

初始提交可以完全没有 `packages/`、TypeScript 源码或自研 Bundle。Harness Runtime 独立安装或作为 CI 制品部署；真实 Profile、Session、Workspace 数据、SQLite、凭证和日志位于部署数据目录。Harness 源码只在需要调试上游时作为并列、可丢弃的 clone，不进入本仓库，也不使用 Git submodule。

### 3.4 何时才修改 Harness

能力补齐的优先级固定为：

1. 调整 Distribution 的 Profile、Patch、prompt、preset 和权限；
2. 启用 Harness 已有 Plugin、Service、Event 和持久化能力；
3. 接入或配置第三方 Bundle；
4. 为现有插件增加窄 adapter，或向其上游贡献修复；
5. 对已证明的稳定缺口实现最小 Secretary 扩展；
6. 只有缺少必要上游 seam 时才向 Harness 贡献修改；
7. 无法上游化且长期必需时，才维护最小 fork。

不得为了符合 v2 的单会话实现而删减 Harness 的 Session、Goal、Workflow 或 Workspace 能力。

## 4. Harness 复用能力与缺口

| 需求 | Harness/生态候选 | Distribution POC 必须回答 | 当前判断 |
|---|---|---|---|
| Workspace 注册 | 官方 Workspace | 通道身份能否稳定映射到 cwd/Workspace，归档后能否恢复 | 待验证，不预设自研 |
| Session | event log、persistence、compaction | Home/Work/Research 是否需要显式分类，现有 Session 是否已足够 | 待验证 |
| 注意力隔离 | preset、system-prompt section | 仅靠配置能否得到共享身份与任务隔离 | 高概率可配置，需回放证明 |
| 跨会话召回 | Session Query、Reference、Memory 插件 | 相关性、权限和上下文上限是否满足秘书场景 | 待验证 |
| 当前目标 | Goal service | 能否覆盖承诺、跟进和重启恢复 | 待验证 |
| 长会话 | Compaction、tool-result pruning | 是否能保持长期协作连续性而不膨胀上下文 | 待验证 |
| 后台工作 | Workflow、Subagents、Jobs、Research 插件 | 并发 1 下能否恢复、产出报告并通知原渠道 | 待验证 |
| 状态存储 | storageDomain、SQLite/JSONL、Memory 插件 | 是否仍需要 Secretary 自有 schema | 不应预设需要 |
| 外部副作用 | Channel/Cron/Webhook 插件、Cordis lifecycle | 幂等、lease、drain 是否已经由组合提供 | 风险较高，优先实测 |

该矩阵的右侧不是“Workspace 插件开发列表”。每一行都必须记录：使用了哪些版本和配置、通过了哪些 v2 契约、失败证据是什么、能否用更小的配置或适配改动解决。只有最后一类稳定缺口才转成代码任务。

Harness 的 Session 是事件源日志，持久化与模型历史投影分离；Compaction 也是 per-session 的事件化过程。这支持“共享项目、隔离 transcript”的设计。参考 [Session subsystem](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session.md) 和 [Compaction subsystem](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/compaction.md)。

官方 per-session preset 架构说明给出的内部测量约为每 Session 组合 12 行配置耗时 3 ms、内存约 600 KB。该数据只说明配置隔离成本低，不代表完整 Agent Session、历史加载或插件组合的总内存。参考 [per-session agent presets](https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/implemented/architecture/2026-08-03-per-session-agent-presets.md)。

## 5. 插件生态接入与副作用治理

### 5.1 候选发现

发现入口包括：

- [GitHub `dsh-plugin` topic](https://github.com/topics/dsh-plugin)
- [DeepSeek Harness Discussions](https://github.com/deepseek-ai/deepseek-harness/discussions)
- [HackSing/dsh-plugins](https://github.com/HackSing/dsh-plugins)
- [dshworks/awesome-dsh-plugins](https://github.com/dshworks/awesome-dsh-plugins)

目录只用于发现，不代表维护、安全、许可证或兼容性背书。搜索时应同时使用协议词和 Harness 能力词，例如 `telegram cordis`、`feishu dsh bundle`、`webhook dsh-plugin`、`cron scheduler deepseek-harness`、`memory workspace dsh`。

### 5.2 当前候选

以下只用于 POC，版本必须重新锁定和审计：

| 需求 | 候选 | 初步策略 |
|---|---|---|
| Telegram + 飞书 | [ZinkLu/dsh-channel](https://github.com/ZinkLu/dsh-channel) | 优先做契约适配，必要时窄 fork |
| 飞书 | [imetn/dsh-lark-bridge](https://github.com/imetn/dsh-lark-bridge)、[xmanrui/dsh-feishu](https://github.com/xmanrui/dsh-feishu) | 备选实现 |
| Telegram | [lovedheart/dsh-plugin-telegram](https://github.com/lovedheart/dsh-plugin-telegram) | 备选实现 |
| Webhook | [ben7am1n/dsh-webhook-bridge](https://github.com/ben7am1n/dsh-webhook-bridge) | 只作为协议底座，补验签、幂等和输入限制 |
| Cron | [csiroqa/dsh-schedule](https://github.com/csiroqa/dsh-schedule) | 验证重启恢复、单实例和主 Workspace 路由 |
| 条件触发 | [fuhefei/dsh-sentinel](https://github.com/fuhefei/dsh-sentinel) | 可复用触发/租约思路 |
| Memory 搜索 | [guntur-d/dsh-memory](https://github.com/guntur-d/dsh-memory)、[NinjaSln-labs/dsh-plugins](https://github.com/NinjaSln-labs/dsh-plugins) | 只能作为索引组件，不能成为 Memory 所有者 |
| 深度研究 | [omdsh-dev/dsh-deep-research](https://github.com/omdsh-dev/dsh-deep-research) | 验证并发 1、恢复、报告落盘和完成通知 |

任何第三方 Memory、Research 或 Channel 插件都先通过 Distribution 契约测试验证。若其公开服务已经满足状态与权限边界，直接配置使用；只有出现无法隔离的耦合时才增加 adapter。不能先设计 Workspace service，再强迫所有插件适配这套尚未证明必要的接口。

### 5.3 “无损加载/卸载”的边界

Cordis Fiber 能回收通过其 API 注册的 listener、service、tool、子插件，以及正确放入 `ctx.effect()` 的 timer、socket、watcher 和进程。参考 [Lifecycle and effects](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cordis-tutorial/02-lifecycle-and-effects.md)。

它不能自动撤销：

- 已发送的 Telegram/飞书消息或 Webhook callback；
- 已提交的数据库事务、文件写入和 schema migration；
- 已产生的模型费用、远端任务和队列消息；
- 未挂入 Fiber 的全局变量、timer、socket 或子进程；
- unload 与收件、Cron、tool call 并发时产生的重复副作用。

因此生产中的状态型、通道型和调度型插件采用 `stop intake -> drain -> flush -> release lease -> dispose -> switch`，不把 HMR 当作发布机制。

### 5.4 准入与持续跟进

每个外部 Bundle 使用以下状态机：

~~~text
discovered -> quarantined -> tested -> accepted -> deprecated -> removed
                                  \-> rejected
~~~

每个 release manifest 至少锁定：Harness tag/commit、Node/pnpm、Bundle 精确版本或 commit、包 integrity、Profile 层顺序、最终配置、权限集合和状态 schema。

兼容性矩阵一次只提升一个变量：

| Lane | Harness | 第三方 Bundle | Secretary Extension | 目的 |
|---|---|---|---|---|
| A Baseline | 当前 | 当前 | 当前 | 证明发布可复现 |
| B Harness-next | 候选 | 当前 | 当前 | 定位上游变化 |
| C Plugin-next | 当前 | 每次一个候选 | 当前 | 定位生态变化 |
| D Extension-next | 当前 | 当前 | 候选或无 | 验证经缺口批准的最小扩展 |
| E Full-next | 候选 | 已通过 C 的组合 | 候选 | 形成下一发布候选 |

CI 每日只生成更新、依赖、安装脚本、权限和 Patch diff；每周跑 B/C/D；人工批准后才进入 E。生产主机不检查或安装社区源码。

## 6. 目标运行配置

### 6.1 Profile 组合

~~~text
secretary-production Profile
├── pinned Harness base/runtime
├── required official Session/Workspace/Compaction bundles
├── selected channel bundles
├── selected scheduler/research bundles
├── optional Secretary extension     # 仅在缺口评审后存在
├── secretary production policy patch
└── machine-local secrets and paths
~~~

Profile 是一次运行的有序 Bundle 组合；Cordis Plugin 是实际代码单元；Bundle 是携带插件代码和配置 Patch 的分发单元。参考 [Harness Architecture](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md) 和 [Package and install a plugin](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/publish.md)。

### 6.2 Session presets

建议初始只启用三种 preset：

| Preset | 用途 | 工具边界 | 并发 |
|---|---|---|---|
| home | 日常对话、收件、分流 | Memory、Task、轻量查询、投递 | 前台 1 turn |
| work | 明确任务 | 按任务显式授权工具 | 全局与 home 共享并发预算 |
| research | 后台研究 | Web/search、受限文件输出，不默认开放 terminal/code runtime | 同时 1 个 run |

完成或长期空闲的 Work/Research Session 应压缩并归档。live Session 数量由运行时预算控制；持久化 Session 数量不等于同时驻留内存的 Session 数量。

### 6.3 数据与持久化

- Harness Session 使用官方持久化实现，不把 Secretary 状态塞入 Session event。
- 先使用组合中既有的 storageDomain/schema；只有契约验证证明需要 Secretary 自有状态时，扩展才建立独立 schema，并只持有被确认的最小数据集。
- 初始优先 SQLite，启用 WAL；prepared Session cache 从小值开始实测。
- `memory.md` 继续是可人工编辑、可备份的事实层；结构化索引用 revision 与其协调。
- 所有 migration 先在生产数据副本验证，明确可回滚、双读或只能前滚。

Harness 的 SQLite Session persistence 默认使用 WAL、写批处理和 prepared Session cache；配置仍需按目标机验证。参考 [session-persistence-sqlite](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/session/session-persistence-sqlite/README.md) 和 [storage-sqlite](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/storage/storage-sqlite/README.md)。

## 7. 2 vCPU / 约 2 GB 内存可行性

### 7.1 调查证据与边界

2026-08-21 对当前目标主机的只读采样结果：

| 项目 | 结果 |
|---|---|
| CPU | 2 vCPU |
| 实际可见内存 | 约 1.6 GiB，并非完整 2 GiB |
| 初始可用内存 | 约 608 MiB |
| Swap | 2 GiB，初始已使用约 737 MiB |
| 根分区余量 | 约 11 GiB |
| 当前 v2 Python RSS 快照 | 约 368 MiB；只代表该次采样 |
| Node | v24.15.0，满足 Harness [开发文档](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/development.md)所列 Node 24 支持范围 |

在 `/tmp` 做了隔离的 `@deepseek-ai/dsh@0.1.1-rc.1` npm 安装观察。安装运行超过 2.5 分钟，单个 npm 进程峰值 RSS 约 729 MiB、CPU 约 83%，随后主动终止；swap 使用量上升到约 985 MiB。该实验测量的是**依赖解析/安装压力，不是 Harness 稳态运行内存**，但足以判定生产机不适合源码安装或完整构建。

目前未发现 Harness 官方提供“完整 Web/Channel/Workflow 组合在 2 GB 主机”的 RSS 保证。因此本文的容量数字是 POC 验收预算，不是上游性能承诺。

### 7.2 资源驱动因素

真正风险不在 Workspace 记录数量本身，而在以下同时驻留工作：

1. 大 Session event log 被完整准备或投影；
2. 未裁剪的大 tool result、网页内容和终端输出；
3. 多个 Agent/Workflow worker 同时运行；
4. code runtime、LSP、持久终端、Chromium 等子进程；
5. 社区插件无界队列、listener、缓存或 write-behind；
6. 本地 Codex/Claude CLI 与 Harness 同机并发；
7. npm/pnpm 源码安装、TypeScript 构建和 Web 打包。

Harness 配置中 Workflow 自动并发在 2 核上会解析为 1，但生产仍应显式设为 1，避免默认值或机器规格变化。每个 Workflow run 还会占用一个 worker thread。参考 [config catalog](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/config-catalog.md) 和 [workflow-worker-thread](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/workflow/workflow-worker-thread/README.md)。

官方仓库的现场问题也说明必须限制长历史和并发：大量 child result 曾触发约 4 GB JS heap OOM；约 20 个 in-process subagent 会影响交互输入；极大或损坏的历史可能在冷加载时阻塞 Web 进程。这些是特定版本/负载的问题报告，不等于当前版本必然复现，但应进入压力测试。参考 [#1897](https://github.com/deepseek-ai/deepseek-harness/discussions/1897)、[#477](https://github.com/deepseek-ai/deepseek-harness/discussions/477) 和 [#1550](https://github.com/deepseek-ai/deepseek-harness/discussions/1550)。

### 7.3 2 GB 生产约束

在资源 POC 证明以前，必须采用以下基线：

- 模型、embedding 和重型研究能力走远程 API，不在本机加载模型；
- 所有 Harness/Bundle 由 CI 构建为不可变 tgz 或镜像，目标机只解包/安装预构建制品；
- Agent/Workflow/Research 全局并发显式设为 1；队列可等待，不并行扩容；
- `maxTotalAgents` 从宽松默认值收紧到小规模保护值，初始建议 8；单个研究任务自行再限子任务数；
- 同时只保留 Home + 1 个前台 Work + 1 个后台 Research 的活跃上限，其他 Session 持久化后释放；
- 保留 Compaction 和默认 tool-result pruner；网页、终端和 callback payload 使用更小的显式上限，超限内容 spill 到文件；
- home/research 默认不加载 code runtime、LSP、持久 terminal、Playwright/Chromium；确需使用时转移到独立 worker；
- 不在该机同时运行本地 Codex/Claude CLI 深度研究；原 v2 这类实现需改为 Harness 受限子任务或外部工作节点；
- Web UI 在用户浏览器渲染，服务器不得为 UI 启动本地浏览器；
- swap 只用于吸收瞬时峰值，持续 swap-in/out 视为容量失败；
- 使用 systemd/cgroup 限制整棵 Harness 进程树，OOM 时优先保住系统和投递账本。

可从以下保护值开始 POC，再根据实测调整，而不是直接作为最终生产参数：

| 保护项 | 当前 1.6 GiB 主机 | 完整 2 GiB 主机 |
|---|---:|---:|
| Node old-space 上限候选 | 512 MiB | 768 MiB |
| Harness cgroup MemoryHigh 候选 | 850 MiB | 1.1 GiB |
| Harness cgroup MemoryMax 候选 | 1.1 GiB | 1.4 GiB |
| Workflow/Agent 并发 | 1 | 1 |
| 同时 Research run | 1 | 1 |

这些限制包含 Harness 的子进程；如果正常功能在限制内频繁 OOM，结论应是组合不适合该主机，而不是依赖 swap 或无限上调边界。

### 7.4 稳定性验收门槛

下列数值是面向当前主机余量设定的工程门槛：

| 场景 | 当前 1.6 GiB 主机目标 | 完整 2 GiB 主机目标 |
|---|---:|---:|
| Harness 完整组合 idle RSS | ≤ 300 MiB | ≤ 350 MiB |
| Home 单 turn 峰值 RSS | ≤ 550 MiB | ≤ 650 MiB |
| Home + 1 Research 峰值 RSS | ≤ 850 MiB | ≤ 1.0 GiB |
| 系统 MemAvailable | 持续 ≥ 250 MiB | 持续 ≥ 300 MiB |

同时必须满足：

1. 24 小时 idle/channel soak 和 72 小时混合工作负载 soak 均无 OOM、异常重启和 handle/RSS 单调增长。
2. 背景 Research 运行时，新 Home 输入在 3 秒内被接纳或明确排队，不能无响应。
3. 1 小时稳态后 swap 使用不持续增长，major page fault 不形成持续尖峰。
4. 反复创建、压缩、归档 Session 后内存能回落到可解释基线。
5. 回放长历史、大网页、大 tool result、重复 Webhook 和网络抖动时，输出上限和 spill 生效。
6. cgroup 触发 MemoryHigh 或受控 OOM 后，Delivery/Cron 账本一致，重启不重复发送。
7. 插件执行 100 次 load/exercise/unload 后，无 listener、timer、socket、worker、子进程和内存持续增长。

### 7.5 Go / No-Go 判断

当前结论是：

- **可以开始**：在 CI 构建预制品；在当前机部署最小 Secretary Distribution 基底；使用 mock channel、远程模型、并发 1 采集基线和能力缺口。
- **暂不可承诺**：完整 Telegram + 飞书 + Webhook + Cron + Research 组合在当前 1.6 GiB 主机长期稳定。
- **明确不可做**：在生产机源码安装/构建；同时运行本地编码 Agent；并行多个 Research/Subagent；用 swap 掩盖持续内存增长。
- **资源 No-Go**：最小完整组合 idle 超过 350 MiB，或单 Research 使 cgroup 超过 1.0 GiB、系统可用内存低于 250 MiB，且无法通过禁用非核心工具解决。

若资源 No-Go，按顺序处理：先移除非核心 Web/开发工具和高开销第三方插件；再把 Research/浏览器/编码任务迁移到独立 worker；仍不满足时升级到 4 GB，而不是削弱 Workspace 的核心语义。

## 8. 分阶段实施计划

实际实现工作的第一步是阶段 1 的开箱可用配置基底。阶段 0 只是从现有行为提取验收标准，不创建 v3 架构或代码。

### 8.1 前置阶段 0：冻结可观察契约

1. 从 v2 提取 Channel、Memory、Task、Scheduler、Research 和 Delivery 行为契约。
2. 契约描述输入、输出、状态、恢复和副作用，不预设必须存在 Workspace 插件、TaskIndex 或特定 schema。
3. 为每项需求定义“完全满足、配置可满足、需适配、缺失、不可接受”五类判定。
4. 建立去敏 replay、fake channel、虚拟时钟和故障注入 fixture。
5. 明确 v2 的真实生产行为与历史偶然实现，避免把单会话等技术取舍误写成需求。

退出条件：每项核心需求都有可执行验收标准，而且不包含预设实现。

### 8.2 阶段 1：建立开箱可用的 Harness 配置基底

1. 锁定一个 Harness tag/commit、Node/pnpm 和最小官方 Bundle 集合。
2. 建立 development、test、production Profile/Patch 模板和 release manifest。
3. 只使用官方 Workspace、Session、Persistence、Prompt、Preset、Goal、Compaction、Query/Reference 等能力启动最小组合。
4. 提供远程模型、目录、SQLite、日志、权限、并发 1 和资源保护的安全默认值。
5. 在 CI 构建不可变制品；目标机只部署预构建产物，不运行源码安装或 Web 构建。
6. 保存最终 dump-config，执行真实 boot、单 turn、重启恢复和资源 smoke。

此阶段不创建 `dsh-secretary-workspace`，也不编写 Secretary 运行时代码。退出条件是得到一套他人可按文档启动、可复现、可测量的 Distribution 基底；若最小基底已经突破第 7 章门槛，立即 Resource No-Go。

### 8.3 阶段 2：逐项组合插件并调查缺口

1. 按“一个 Channel -> Cron -> 第二 Channel -> Webhook -> Research -> 可选 Memory”顺序，每次只增加一个外部 Bundle。
2. 所有候选先做许可证、安装脚本、依赖、权限、持久化和生命周期审计，再在隔离环境运行 Lane C。
3. 对每种组合运行阶段 0 的契约、重启恢复、重复输入、卸载和去敏 replay。
4. 先尝试 Profile/Patch、prompt、preset、权限、官方服务和插件自身配置，不写 adapter。
5. 记录每项能力由哪个 Bundle/配置拥有、状态写在哪里、卸载后留下什么、升级和回滚边界是什么。
6. 形成 `capability-matrix.md`、`gap-analysis.md` 和 ADR，而不是直接创建开发任务。

每个缺口必须归类：

| 类别 | 含义 | 动作 |
|---|---|---|
| A | Harness 官方能力已经满足 | 固化配置和测试 |
| B | 第三方插件已经满足 | 锁定版本、权限和契约 |
| C | 只缺配置转换或窄协议适配 | 优先上游贡献或实现最小 adapter |
| D | 多项需求共享稳定的 Workspace 语义缺口 | 才评估 Secretary Workspace Extension |
| E | 组合在安全、资源或维护上不可接受 | 更换插件、外置能力或判定方案 No-Go |

退出条件：每项 v2 核心契约都有证据和分类。如果全部落入 A/B，直接跳过阶段 3，自研 Workspace 工作量为零。

### 8.4 阶段 3：条件式最小扩展

本阶段只有存在 C/D 类缺口且 ADR 获批时才执行：

1. 每个扩展必须引用失败的契约、已尝试的配置/插件方案和不采用上游适配的理由。
2. C 类优先实现单一 adapter，不引入新的中心状态和抽象层。
3. 只有多个 D 类缺口确实共享身份、记忆、任务索引或注意力生命周期时，才聚合为 `dsh-secretary-workspace` Bundle。
4. 只实现通过测试所需的最小接口；不预先实现完整 workspace-registry、memory、task-index、router 或 delivery 框架。
5. 新状态必须有明确 owner、schema、migration、备份和降级策略。
6. 扩展完成后重新运行 A-E 兼容性矩阵，证明它补的是稳定需求而不是掩盖插件配置问题。

退出条件：所有剩余契约通过；新增代码量、状态面和权限面都有缺口证据支撑。若插件组合已经满足需求，本阶段不存在。

### 8.5 阶段 4：资源与副作用验证

1. 运行 24 小时 idle/channel soak。
2. 运行 72 小时混合负载：日常对话、Cron、一个 Research、长历史和网络故障。
3. 注入重复收件、插件 unload、进程 kill、磁盘压力、API 429 和 schema 升级失败。
4. 检查 RSS、heap、FD、worker、socket、swap、事件循环延迟和投递账本。
5. 分别验证当前 1.6 GiB cgroup 与完整 2 GiB 预算；记录每个 Bundle 的增量成本。
6. 删除或外置无法解释的高开销能力，但不能破坏阶段 0 的需求契约和状态正确性。

退出条件：全部满足第 7.4 节；任何一次重复发送、账本损坏或跨 Session 污染都阻止进入 canary。

### 8.6 阶段 5：影子运行与有限 canary

- 使用独立 DSH_HOME、Workspace、Session、SQLite、测试 token/端口和虚拟 Cron；
- replay 去敏的 v2 输入，比较状态变化、拟发送消息和资源曲线；
- 双写输入时只有一个系统持有真实外发 lease；
- 先迁移只读工具和单一通道收件，再切换主动发送、Webhook 消费和 Cron lease；
- 切换前 drain v2 在途 turn、投递和研究任务，切换点写入审计账本；
- canary 期间限制用户、通道和任务类型，并保留上一 release manifest 和数据备份。

### 8.7 阶段 6：正式发布和持续演进

1. 归档 Harness Runtime、全部 Bundle tgz、integrity、release manifest、最终配置和测试证据。
2. 生产只从已归档制品恢复，不追踪 `master`、npm `latest` 或浮动 Git dependency。
3. 每日更新雷达只报告变化；每周运行单变量 Lane B/C；只有存在自研扩展时才运行 Lane D，通过后再形成 Full-next。
4. 新版本先在影子 Profile 运行，再 drain-and-swap；不得依赖业务副作用可被 unload 撤销。
5. 每月复核 fork 数量、兼容性滞后、内存增量和第三方维护活跃度。
6. 若连续两个候选周期无法满足资源或兼容门槛，重新评估外置 worker、4 GB 主机或其他底座。

### 8.8 v3 总体 Go 条件

只有同时满足以下条件，才从 v2 正式迁移：

1. Distribution 对阶段 0 的全部核心需求给出可执行配置和通过证据；Workspace 行为可以由官方、第三方或经批准的最小扩展提供。
2. 至少一个生产 Channel、Cron 和 Delivery 组合证明单实例、幂等和 drain。
3. Research 在并发 1 下可恢复或可靠收敛为失败，不长期停留在 running。
4. Harness、所有外部 Bundle 和任何可选 Secretary Extension 都有精确版本、integrity、权限清单和回滚制品。
5. 目标机通过第 7.4 节的 24/72 小时资源门槛。
6. 去敏 replay 和 canary 无重复外发、丢失提醒、状态损坏或跨 Workspace/Session 污染。
7. 所有自研代码都能回溯到 C/D 类缺口；如果不存在这些缺口，正式制品中不包含 Secretary 运行时代码。

在这些条件满足前，Secretary v2 保持生产基线；验证 Harness 不应改变现有服务控制、生产 token、Webhook、数据库或外发 lease。
