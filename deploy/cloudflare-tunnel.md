# Cloudflare Tunnel 部署指南（大陆 VPS + xray 借道）

将 Secretary 的 HTTP webhook（`127.0.0.1:11269`）通过 Cloudflare Tunnel + Zero Trust
暴露到公网，适用于直连 Cloudflare edge 被干扰的大陆机房（阿里云等）。

实测环境：Ubuntu 22.04，xray 本地 HTTP 代理 `127.0.0.1:10802`，
cloudflared 2026.2.0（Zero Trust 网页端 token 安装方式）。

## 背景结论（实测，勿重走弯路）

| 尝试 | 结果 |
|---|---|
| cloudflared 默认 QUIC（UDP 7844） | 握手黑洞，`failed to dial to edge with quic: timeout` |
| 强制 http2（TCP 7844）直连 | TCP 三次握手成功，但 TLS ClientHello 后被 SNI 级 DPI 黑洞 |
| `http_proxy` / `https_proxy` / `ALL_PROXY` 环境变量 | cloudflared 的 edge 连接**不读任何代理变量**（前台实验实证，报错五元组仍为本机直连；官方 issue #87 / #1025 长期 open） |
| proxychains4 | 无效——cloudflared 是**静态链接**二进制，LD_PRELOAD 劫持不了 |
| **redsocks + iptables 内核层重定向** | ✅ 唯一生效方案 |

最终链路：

```
Internet → Cloudflare edge
              ↑ http2 tunnel (TCP 7844)
          cloudflared ──(iptables nat REDIRECT dport 7844)──→ redsocks:12345
                                                                 ↓ http-connect
                                                             xray:10802 → 境外出口
          cloudflared ──(回源，不经代理)──→ 127.0.0.1:11269 (secretary HTTP channel)
```

无回环：iptables 只劫持目标端口 7844；redsocks→xray 走 10802、xray 出站走自己的
服务器端口，均不匹配规则；回源到 localhost 不出网卡。

## 前提

- xray 已运行且本地 HTTP 代理可用（端口按实际调整，下文统一用 10802）：

  ```bash
  curl -x http://127.0.0.1:10802 -sI --connect-timeout 5 https://www.cloudflare.com | head -2
  # 预期: HTTP/1.1 200 Connection established
  ```

- Zero Trust 面板已创建 Tunnel，并配置 Public hostname
  （例：`os.example.com`，Path `^/hooks`，Service `http://localhost:11269`）。

## 步骤

### 1. 安装 cloudflared（网页端 token 指令）

Zero Trust → Networks → Tunnels → 选择 Debian/Ubuntu，复制面板给出的指令，形如：

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared.deb
sudo cloudflared service install <TOKEN>
```

此时服务会反复失败——直连被墙，属预期，继续下一步。

### 2. redsocks（代理桥）

```bash
sudo apt-get install -y redsocks
sudo tee /etc/redsocks.conf >/dev/null <<'EOF'
base {
    log_debug = off;
    log_info = on;
    log = "syslog:daemon";
    daemon = on;
    user = redsocks;
    group = redsocks;
    redirector = iptables;
}

redsocks {
    local_ip = 127.0.0.1;
    local_port = 12345;
    ip = 127.0.0.1;
    port = 10802;
    type = http-connect;
}
EOF
sudo systemctl restart redsocks
ss -tlnp | grep 12345    # 必须看到 redsocks 在监听
```

（若 xray 提供 socks5 端口，`type` 可改 `socks5` 并换端口，更稳。）

### 3. iptables 定向劫持 + 持久化

```bash
sudo iptables -t nat -A OUTPUT -p tcp --dport 7844 -j REDIRECT --to-ports 12345
sudo apt-get install -y iptables-persistent   # 弹窗选 Yes 保存 IPv4 规则
# 以后规则变动: sudo netfilter-persistent save
```

### 4. systemd override（协议 + 超时 + 启动依赖）

原 unit `TimeoutStartSec=15` 太短，`Type=notify` 在重试中就会被杀。
不要用 `systemctl edit`（内容必须写在 "Edits below this comment will be
discarded" 之上，写错位置会被静默丢弃），直接写文件：

```bash
sudo mkdir -p /etc/systemd/system/cloudflared.service.d
sudo tee /etc/systemd/system/cloudflared.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment=TUNNEL_TRANSPORT_PROTOCOL=http2
TimeoutStartSec=180
RestartSec=10s

[Unit]
After=xray.service redsocks.service
Wants=xray.service redsocks.service
EOF
sudo systemctl daemon-reload && sudo systemctl restart cloudflared
```

`TUNNEL_TRANSPORT_PROTOCOL=http2` 必须保留：redsocks 只能接管 TCP，QUIC/UDP 无法借道。

### 5. 验证

```bash
# VPS 上：预期 4 条注册，location 为境外节点（lax/sjc/hkg 等）
journalctl -fu cloudflared | grep -E "Registered|ERR"
# 例: INF Registered tunnel connection connIndex=0 ... location=lax08 protocol=http2

# 确认 override 已生效
systemctl cat cloudflared | tail -10

# 外部机器上：全链路（返回 401/404/405 均说明已穿透到 secretary）
curl -sI https://os.example.com/hooks
```

## 故障排查

| 症状 | 原因 | 动作 |
|---|---|---|
| 报错五元组为 `本机IP:xxx -> 198.41.x.x:7844` 直连 | iptables 规则丢失（常见于重启后未持久化） | 重加第 3 步规则并 `netfilter-persistent save` |
| `failed to dial ... with quic: timeout` | http2 未生效（override 没挂上） | `systemctl cat cloudflared` 核对，daemon-reload 重启 |
| `start operation timed out` 15 秒被杀 | override 的 `TimeoutStartSec=180` 未生效 | 同上 |
| `connection with edge closed` / TLS i/o timeout | xray 或 redsocks 挂了 | 依次 `systemctl status xray redsocks`，再看 7844 规则 |
| 隧道通但外部 404 | Public hostname 的 Path 不匹配 | 面板核对 Path 正则（如 `^/hooks`）与请求路径 |
| 外部 `HTTP 000` / TLS `SSL_ERROR_SYSCALL`，`dig` 无结果 | 面板加 Public hostname 时 DNS 记录未建成 | DNS 页手动加 Proxied CNAME：`<sub> → <tunnel-UUID>.cfargotunnel.com`（必须橙色云）；注意 HTTP 代理的 CONNECT 200 是乐观应答，不代表远端可达 |
| systemd 拒绝再启动（start-limit-hit） | 失败次数烧完重启预算 | 修复根因后 `systemctl reset-failed cloudflared` 再 start |

## 注意事项

- 日志里 `ping_group_range` / `ICMP proxy ... disabled` 的 WRN 无害，可忽略。
- 换 token 或重装 tunnel 后 `cloudflared service install` 会重写主 unit，
  但 override.conf 独立存在、自动继续生效——这是用 drop-in 而非直接改 unit 的原因。
- 所有入站 webhook 流量多一跳 xray；隧道为端到 edge TLS，代理只见密文。
- secretary 本体（Telegram/Feishu/LLM）全部为出站直连，不依赖此隧道；
  隧道离线只影响 webhook 入站。
