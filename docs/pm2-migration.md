# Secretary Agent 从 systemd 迁移到 PM2 的评估与实施方案

> 状态：建议方案，尚未批准实施
>
> 更新时间：2026-07-31
>
> 当前建议：生产环境继续以 systemd 为默认进程管理器；只有在需要统一 PM2 运维入口时，才增加 PM2 可选后端。
>
> 核心原则：迁移不能降低单实例约束、启动失败保护、内存保护、日志保留和数据库维护安全性。

## 1. 结论

Secretary Agent 是单个长驻 Python 进程，PM2 可以通过指定外部虚拟环境的 Python 解释器运行它，并覆盖启动、停止、重启、状态查看、日志查看和异常重启等基本能力。

但不建议仅为了替换 `systemctl` 而全面迁移，原因如下：

1. 在 Ubuntu 上，`pm2 startup` 通常仍会生成 `pm2-<user>.service`，由 systemd 负责开机启动 PM2 daemon。因此迁移可以让日常操作改用 `pm2`，但不会真正消除 systemd。
2. 当前 unit 使用 cgroup `MemoryHigh` 和 `MemoryMax` 限制 Secretary 及其 Claude/Codex 子进程。PM2 的 `max_memory_restart` 是周期检查后的重启策略，不是等价的硬内存上限。
3. 当前 unit 还提供 `NoNewPrivileges`、`PrivateTmp`、网络就绪依赖、有限失败重启和停止超时。迁移时必须明确保留或接受这些能力的变化。
4. 当前部署脚本、数据库重建脚本、日志轮转和运维说明都直接依赖 systemd，迁移范围不只是增加一个 ecosystem 文件。
5. 项目包含 Telegram polling，任何新旧进程并行都会产生重复消费或 Telegram `409`。切换必须是严格的单实例停机切换。

因此推荐按以下优先级决策：

1. 仅希望简化命令：保留 systemd，增加统一的 `secretaryctl` 管理入口。
2. 希望一套部署同时支持多种主机：保留 systemd 默认后端，增加 PM2 可选后端。
3. 服务器已有统一 PM2 监控和运维体系：实施 PM2 后端，并通过生成的 systemd unit 保留开机自启和必要的 cgroup 安全边界。

## 2. 迁移目标与非目标

### 2.1 目标

- 使用普通部署用户执行 `pm2 status`、`pm2 logs secretary`、`pm2 restart secretary` 等日常操作。
- 保持唯一一个 `main.py --channel all` 进程。
- 保持当前外部虚拟环境，不在源码目录内新建或绑定运行环境。
- 启动自检失败时退出并有限重试，避免无限消耗模型配额或重启预算。
- 停止时给应用足够时间关闭调度器、渠道连接和后台子进程。
- 保持 HTTP webhook 只监听 `127.0.0.1:11269`，不改变 Cloudflare Tunnel 拓扑。
- 保持日志大小、保留周期和数据库重建流程可验证、可回滚。
- 支持从 PM2 快速回滚到原 systemd unit。

### 2.2 非目标

- 不使用 PM2 cluster mode，也不启动多个 Secretary 实例。
- 不使用 `watch` 自动重启；代码或日志变化不能触发生产进程重启。
- 不把 PM2 当作 Python 依赖管理器；Python、虚拟环境和依赖仍由 `uv` 管理。
- 不用 PM2 暴露新的公网端口。
- 不在迁移中改动 Telegram、飞书、HTTP、调度器或数据库业务逻辑。
- 不同时保留 systemd 和 PM2 对同一应用的启动权。

## 3. 当前 systemd 基线

当前 `deploy/templates/secretary.service` 提供以下行为：

| 当前能力 | 配置 | 迁移要求 |
|---|---|---|
| 工作目录 | `secretary_v2/` | PM2 `cwd` 必须一致 |
| Python 解释器 | 外部 `@VENV_DIR@/bin/python` | PM2 必须使用绝对路径 |
| 启动参数 | `main.py --channel all` | 保持不变 |
| 异常重启 | `Restart=on-failure` | 正常退出不重启，异常退出才重启 |
| 重启间隔 | `RestartSec=15` | `restart_delay: 15000` |
| 失败预算 | 10 分钟最多失败 5 次 | 用 `min_uptime` 和 `max_restarts` 近似，并实测语义 |
| 停止超时 | `TimeoutStopSec=20` | `kill_timeout: 20000` |
| 网络依赖 | `network-online.target` | 由 PM2 startup unit drop-in 保留 |
| 环境变量 | 明确设置 `HOME`、`PATH` | ecosystem 中显式设置 |
| 日志 | stdout/stderr 合并到固定文件 | 保持固定路径，避免破坏告警和自检脚本 |
| 内存保护 | `MemoryHigh=1300M`、`MemoryMax=1600M` | PM2 无直接等价能力，需保留 cgroup 约束或接受降级 |
| 安全隔离 | `NoNewPrivileges=true`、`PrivateTmp=true` | 通过 PM2 startup unit drop-in 评估保留 |

相关实现：

- systemd unit：[`deploy/templates/secretary.service`](../deploy/templates/secretary.service)
- 部署脚本：[`deploy/deploy.sh`](../deploy/deploy.sh)
- 数据库维护：[`deploy/rebuild-database.sh`](../deploy/rebuild-database.sh)
- 日志轮转：[`deploy/templates/logrotate-secretary`](../deploy/templates/logrotate-secretary)
- 日志清理 timer：[`deploy/templates/secretary-logclean.timer`](../deploy/templates/secretary-logclean.timer)

## 4. PM2 目标配置

实施时建议增加模板 `deploy/templates/ecosystem.config.cjs`，由部署脚本渲染 `@APP_DIR@`、`@APP_HOME@` 和 `@VENV_DIR@`。概念配置如下：

```javascript
module.exports = {
  apps: [
    {
      name: "secretary",
      cwd: "@APP_DIR@/secretary_v2",
      script: "main.py",
      interpreter: "@VENV_DIR@/bin/python",
      args: ["--channel", "all"],

      instances: 1,
      exec_mode: "fork",
      watch: false,

      autorestart: true,
      stop_exit_codes: [0],
      restart_delay: 15000,
      min_uptime: "10m",
      max_restarts: 5,
      kill_timeout: 20000,

      env: {
        HOME: "@APP_HOME@",
        PATH: "@APP_HOME@/.local/bin:/usr/local/bin:/usr/bin:/bin"
      },

      out_file: "@APP_DIR@/secretary_v2/logs/secretary_v2.log",
      error_file: "@APP_DIR@/secretary_v2/logs/secretary_v2.log",
      merge_logs: true,
      time: false
    }
  ]
};
```

配置说明：

- `stop_exit_codes: [0]` 用于接近 systemd `Restart=on-failure` 的语义。启动自检失败会以非零状态退出并进入重试，正常退出不会被 PM2 自动拉起。
- `min_uptime: "10m"` 与 `max_restarts: 5` 只是对当前 systemd 失败预算的近似映射。PM2 统计的是连续“不稳定重启”，systemd 统计的是时间窗口内的启动次数，实施前必须用故障注入验证。
- `watch` 必须关闭。仓库中的数据库、日志、memory 文件和运行状态会持续变化，启用 watch 会导致意外重启。
- 不使用 cluster/reload 零停机能力。Telegram polling、飞书连接、内置调度器和本地 SQLite 都要求单实例。
- `time: false` 避免 PM2 在已有 Python 日志时间戳外重复添加时间戳。
- 不应把 `max_memory_restart` 当作 `MemoryMax` 的替代品。若额外配置它，只能作为辅助重启阈值。

## 5. systemd 仍然保留的职责

如果使用 PM2 的标准开机自启方案，目标链路将变为：

```text
systemd
  └── pm2-<user>.service
        └── PM2 daemon
              └── Secretary Python process
                    └── Claude/Codex 等子进程
```

这意味着 PM2 迁移的真实收益是统一应用层管理命令和监控入口，而不是移除 systemd。

为避免安全性明显下降，应评估为 `pm2-<user>.service` 增加 drop-in：

- `Wants=network-online.target`
- `After=network.target network-online.target`
- `NoNewPrivileges=true`
- `PrivateTmp=true`
- 适合 PM2 daemon 额外开销的 `MemoryHigh` 和 `MemoryMax`

注意：这些设置会作用于该 PM2 daemon 管理的全部应用。若同一 PM2 daemon 还运行其他服务，资源限制和安全隔离将不再能只针对 Secretary。此时不应直接照搬当前数值，应该继续使用独立 systemd unit，或为 Secretary 使用独立 PM2 home/service 并完成压力测试。

## 6. 代码与文档改动范围

### 6.1 部署脚本

建议将 `deploy/deploy.sh` 拆分为公共配置与进程管理器后端，默认值保持 systemd：

```text
SECRETARY_PROCESS_MANAGER=systemd  # 默认
SECRETARY_PROCESS_MANAGER=pm2     # 显式选择
```

公共步骤继续负责：

- swap 和 swappiness；
- `uv`、Python 和虚拟环境；
- Python 依赖安装；
- `config.yaml` 初始化与占位符检查；
- 日志目录创建；
- 可选测试；
- 启动后健康检查。

PM2 后端新增负责：

- 检查 Node.js、npm 和 PM2，但不静默覆盖现有 Node 运行环境；
- 渲染 ecosystem 文件；
- 启动或重启唯一的 `secretary` 应用；
- 等待 `Startup self-test: PASSED`；
- 执行 HTTP `/health` 检查；
- `pm2 save` 保存当前进程清单；
- 一次性生成并安装 PM2 startup unit；
- 输出 PM2 运维命令。

### 6.2 统一服务控制接口

建议增加 `deploy/secretaryctl`，由部署时保存的后端选择决定调用 systemd 或 PM2：

```text
secretaryctl start
secretaryctl stop
secretaryctl restart
secretaryctl status
secretaryctl logs
secretaryctl health
```

数据库维护脚本只调用这个稳定接口，不再直接拼接 `systemctl` 或 `pm2` 命令。接口必须能够：

- 判断服务是否存在、运行、停止或失败；
- 记录维护前是否正在运行；
- 失败时恢复原运行状态；
- 等待启动自检通过；
- 保留 `--no-start` 语义。

### 6.3 日志

推荐第一阶段继续把应用 stdout/stderr 写入 `secretary_v2/logs/secretary_v2.log`，保留现有 logrotate 规则。这样可以继续复用：

- 启动自检日志检测；
- 数据库重建后的启动验证；
- 现有人工排障命令；
- 现有 20 MB、每日、14 份压缩保留策略。

PM2 自身还会产生 daemon 日志，必须单独限制。可以选择 PM2 官方提供的原生 logrotate 配置或 `pm2-logrotate`，但不能让两套工具同时轮转同一应用日志。

若最终希望移除 `secretary-logclean.timer`，应先确认 logrotate 已覆盖所有新旧文件命名，并对历史 `manage.sh` 时间戳日志执行一次受控清理。不要在迁移切换阶段同时改变日志路径和保留策略。

### 6.4 数据库重建

`deploy/rebuild-database.sh` 当前直接检查、停止和启动 `secretary.service`。迁移后必须通过统一服务控制接口完成以下不变量：

1. 重建前确认服务状态可识别。
2. 服务运行时先停止，并确认进程完全退出。
3. 归档和重建失败时，使用未变更或已恢复的数据库恢复原服务。
4. 成功后仅在维护前服务正在运行且未指定 `--no-start` 时启动。
5. 启动后等待 `Startup self-test: PASSED`，不能只以 PM2 显示 `online` 为成功。

## 7. 实施阶段

### 阶段 0：决策门槛

只有满足以下至少一项才进入实施：

- 主机已有多个 PM2 服务，需要统一管理；
- PM2/PM2.io 监控是明确的运维要求；
- systemd 不是目标平台的稳定公共能力，而 PM2 是；
- 团队愿意承担 Node.js、npm、PM2 daemon 和 startup unit 的维护成本。

如果唯一问题只是命令难记或需要 `sudo`，优先实现 `secretaryctl`，不切换进程管理器。

### 阶段 1：离线准备

1. 增加 ecosystem 模板和 PM2 后端，但保持 systemd 默认。
2. 增加统一服务控制接口。
3. 改造数据库重建脚本使用统一接口。
4. 增加部署脚本静态检查和 PM2 配置解析检查。
5. 更新 README，明确主机只能选择一个进程管理器。

### 阶段 2：影子验证

不要在生产 Telegram token 和端口上并行启动第二实例。影子验证应使用独立配置，至少做到：

- 禁用 Telegram 和飞书，或使用测试凭证；
- HTTP 使用不同端口或完全禁用外部流量；
- SQLite 使用测试数据库；
- 验证启动自检、正常停止、异常退出、失败预算和日志轮转；
- 验证 PM2 发出的 SIGINT 能让 Python `finally` 清理调度器、渠道和子进程。

### 阶段 3：生产切换

生产切换前必须先查看服务状态和日志尾部，不能对未知故障状态盲目重启。

建议顺序：

1. 记录当前 systemd 状态、进程 PID、端口监听和日志偏移。
2. 停止 `secretary.service`，等待进程完全退出。
3. 确认 `127.0.0.1:11269` 无旧监听，且不存在旧 Telegram polling 进程。
4. 使用 ecosystem 文件启动 PM2 中的 `secretary`。
5. 等待日志出现 `Startup self-test: PASSED`。
6. 验证 `/health`、Telegram、飞书和 HTTP webhook。
7. 验证内置 scheduler 已启动且没有重复任务。
8. 执行 `pm2 save`。
9. 安装并验证 PM2 startup unit。
10. 最后才禁用旧 `secretary.service` 的开机启动，但保留 unit 文件用于回滚。

切换过程中任何一步失败，都应先停止并删除 PM2 中的 `secretary`，再恢复 systemd。绝不能为了减少停机时间而同时运行两套生产实例。

### 阶段 4：观察期

至少观察一个完整调度周期，重点检查：

- 无 Telegram `409` 或重复消息；
- 飞书 WebSocket 能稳定重连；
- HTTP health 和 Cloudflare Tunnel 正常；
- 定时任务只执行一次；
- 启动自检失败不会无限重启；
- Claude/Codex 子进程能在停止时被回收；
- PM2 daemon 与所有子进程的合计内存不会突破 VPS 安全余量；
- 应用日志与 PM2 daemon 日志都按预期轮转。

观察期结束后再考虑删除旧 unit；更稳妥的做法是长期保留模板和 systemd 后端。

## 8. 回滚方案

回滚必须保持单实例：

1. 查看应用日志，确认回滚原因。
2. `pm2 stop secretary`，等待 Python 和子进程退出。
3. `pm2 delete secretary`，防止 daemon 再次拉起它。
4. `pm2 save`，更新开机恢复清单。
5. 确认端口和 Telegram polling 进程已经消失。
6. 重新启用并启动 `secretary.service`。
7. 等待 `Startup self-test: PASSED` 并验证 `/health`。
8. 验证 Telegram、飞书、Webhook 和 scheduler。

回滚不修改数据库格式，因此正常情况下不需要数据迁移。若 PM2 运行期间发生数据库维护，仍应使用同一份完整归档验证后再恢复服务。

## 9. 验收标准

只有全部满足才认为迁移成功：

- `pm2 status` 中仅有一个 `secretary` 实例。
- 系统进程中仅有一个 `main.py --channel all`。
- 启动日志出现 `Startup self-test: PASSED`。
- `http://127.0.0.1:11269/health` 正常。
- Telegram、飞书和 HTTP webhook 各完成一次端到端验证。
- scheduler 在一个完整周期内没有漏执行或重复执行。
- 正常退出码 `0` 不自动重启，非零退出会延迟重启。
- 连续启动失败达到预算后，PM2 将应用标为 errored，不再无限重试。
- 停止命令能在 20 秒内完成清理；超时后才强制结束。
- 主机重启后只恢复 PM2 版本的 Secretary，不恢复旧 systemd 实例。
- 应用及子进程的内存边界经过验证，2 GB VPS 不会因 PM2 迁移失去保护。
- 应用日志和 PM2 daemon 日志均有明确且唯一的轮转策略。
- `deploy/rebuild-database.sh --yes` 和 `--no-start` 在 PM2 后端行为正确。
- 回滚到 systemd 的演练成功。

## 10. 风险清单

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| systemd 与 PM2 同时启动 | Telegram 409、重复提醒、SQLite 并发风险 | 单实例检查、严格停机切换、最后才修改开机启动 |
| PM2 内存阈值不覆盖子进程硬上限 | 2 GB VPS OOM | 保留 cgroup 限制、压力测试、必要时继续使用 systemd |
| PM2 默认对正常退出也重启 | 无渠道或显式停止后被重新拉起 | `stop_exit_codes: [0]` 并做故障注入 |
| `min_uptime/max_restarts` 语义不同 | 重试过多或过早进入 errored | 自动化测试并记录恢复命令 |
| PM2 SIGINT 与当前 SIGTERM 行为不同 | 清理不完整、子进程残留 | 影子环境测试 graceful shutdown |
| 应用日志与 PM2 日志重复轮转 | 日志丢失或磁盘增长 | 每个日志文件只指定一种轮转方案 |
| Node/PM2 升级改变启动路径 | 重启后无法恢复 | PM2 升级后重新生成 startup unit 并验证重启 |
| PM2 daemon 管理多个应用 | cgroup 限制无法只约束 Secretary | 独立 PM2 home/service 或保留独立 systemd unit |
| 数据库维护脚本仍调用 systemctl | 维护失败或错误恢复 | 先完成统一服务控制层再切换生产 |

## 11. 待决策项

实施前需要明确：

1. 迁移动机是降低 `sudo/systemctl` 使用成本，还是接入统一 PM2 运维体系？
2. PM2 是否只管理 Secretary，还是与其他应用共享 daemon？
3. 是否接受 PM2 标准 startup 方案仍依赖 systemd？
4. 现有 1.3 GB/1.6 GB 内存阈值在加入 PM2 daemon 开销后如何调整？
5. 日志继续使用系统 logrotate，还是统一迁移到 PM2 日志模块？
6. 是否要长期维护 systemd 与 PM2 两个后端？
7. 是否先只实现 `secretaryctl`，观察运维体验后再决定迁移？

## 12. 参考资料

- [PM2 Startup Script](https://pm2.keymetrics.io/docs/usage/startup/)
- [PM2 Ecosystem File](https://pm2.keymetrics.io/docs/usage/application-declaration/)
- [PM2 Restart Strategies](https://pm2.keymetrics.io/docs/usage/restart-strategies/)
- [PM2 Graceful Start/Shutdown](https://pm2.keymetrics.io/docs/usage/signals-clean-restart/)
- [PM2 Memory Limit Reload](https://pm2.keymetrics.io/docs/usage/memory-limit/)
- [PM2 Log Management](https://pm2.keymetrics.io/docs/usage/log-management/)
