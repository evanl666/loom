# 🧵 Loom

[![PyPI](https://img.shields.io/pypi/v/loom-harness)](https://pypi.org/project/loom-harness/)
[![CI](https://github.com/evanl666/loom/actions/workflows/ci.yml/badge.svg)](https://github.com/evanl666/loom/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/loom-harness)](https://pypi.org/project/loom-harness/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

<p align="right"><a href="README.md">English</a> · <b>中文</b></p>

### AI Agent 的黑匣子、防火墙和调试器。

你的 agent 跑完了——碰了文件、调了工具、烧了 token——但你根本不知道它干了啥、为啥这么干。
**Loom 记录它的每一个动作,以 $0 逐字节回放,在危险调用执行前把它拦下来,还让你像用调试器一样单步走完整个运行过程。**
支持任何 Claude / OpenAI-API agent——Claude Code、LangGraph、CrewAI、你自己写的都行。

```bash
pip install loom-harness          # 零依赖
loom record claude "fix the failing test" --safe
```
```
recorded 17 steps · 42k tokens → session.loom.json
🛡  firewall blocked 1 risky call:  Read(".env")
🔬 loom debug session.loom.json   # 单步走一遍,任意一步都能 fork 重跑
```

---

## 为什么用 Loom

- 🎥 **录制任何 agent** —— 代理 Claude Code / Codex / Cursor / 你自己的,一条命令,零代码改动。
- ⏪ **$0 回放** —— 每次调用都在同一个边界录下 → **逐字节一致、离线**。给随机的 agent 做确定性 CI。
- 🔬 **单步调试** —— 逐步查看,看到**模型当时真实看到的上下文**,然后**改某一步、live 重跑**。
- 🕸 **任意多智能体框架** —— LangGraph · CrewAI · AutoGen · OpenAI-Agents · Claude-SDK,从线路上还原成一棵 **agent 树**,零代码改动。
- 🛡 **防火墙** —— 在危险调用**执行前**拦截 / 确认,按能力(`cap:money_movement`)或按序列(`读了 .env 之后:禁止联网`)。
- 🕵 **抓数据外泄** —— 密钥从读取流向出网,哪怕**被 base64 编码或改写过**,由 LLM 判定确认。
- ↩ **撤销世界状态** —— 回滚 agent 改过的文件,或快照 & 恢复整个工作区 + 数据库。

---

## 调试器

`loom debug run.loom.json`(或用 `loom live` 边跑边看)在浏览器里打开一个单步调试器:

- **单步**走每个动作 —— 模型的推理、工具调用 + 参数、世界差异(文件 / SQL 行 / DOM)、风险、token。
- **上下文帧** —— 模型在这一步真实看到的对话:相当于调试器的**调用栈 & 变量**。
- **Fork & live 重跑** —— 在任意一步注入一条消息或换个模型;只有分叉后的尾部才产生调用,新分支就并排显示在原始运行旁边。
- **多智能体树** —— 主管/子 agent 系统(你自己的或第三方框架的)从线路还原,以可折叠的树显示,按 agent 分道着色。
- **发消息 & 断言** —— 给 live agent 发一条新消息,或用大白话写期望(`never issue_refund`、`output contains …`)当 CI 门禁。

`loom studio <trace>` 把整个界面冻成**一个可分享的 HTML 文件**(不用服务、不用 agent)。

---

## 调试一个 live agent

```bash
loom live --agent app:agent        # 边跑边看、发追问、任意一步 fork
```

agent 藏在 **gRPC / HTTP 端点**后面?把你的 server 指向录制代理,在同一个调试器里驱动它——不用写代码,直接用你的 `grpcurl`:

```bash
loom live --proxy-port 9000 \
  --trigger 'grpcurl -d "{\"prompt\": $LOOM_PROMPT_JSON}" -plaintext :50051 agent.Agent/Run'
# 然后用 ANTHROPIC_BASE_URL=http://127.0.0.1:9000 启动你的 server
```

哪怕 agent 藏在端点后面,Loom 也能还原它**内部完整的多层结构**。

---

## 当作 Python harness 用

```python
from loom import Agent, tool, Policy

@tool
def search(q: str) -> str:
    "Search the docs."
    return db.search(q)

agent = Agent(model="claude-opus-4-8", tools=[search],
              policy=Policy(deny=["issue_refund*"], budget_tokens=50_000))  # 回路内防火墙
run = agent.run("What changed in the API last week?")

run.replay()        # 逐字节一致,不发任何 API 调用
run.fork(at=3)      # 倒回第 3 步,在新分支上继续 live
```

一个**效应边界(effect boundary)**记录每一次模型 + 工具调用 —— 于是回放、fork、免费 CI、
人在回路、防火墙、以及每一个分析器,都从同一个原语里长出来。内核**零依赖**。

---

## 更多命令(节选)

| | |
|---|---|
| `loom replay <trace>` | 逐字节重跑,$0,离线 |
| `loom taint` · `loom dlp --judge` | 外泄血缘 · 语义 DLP |
| `loom redteam run --generate <m>` | AI 红队 —— 针对**你的**工具面自动生成攻击 |
| `loom mcp gateway -- <server>` | 给任意 MCP server 加防火墙 + 录制 |
| `loom undo <trace>` | 回滚 agent 改过的文件 |

运行 `loom --help` 看全集 —— record、replay、debug、live、studio、firewall、
taint、dlp、redteam、mcp、undo、cost、rootcause、experiment 等等。

---

## 安装

```bash
pip install loom-harness                # 内核 + CLI,零依赖
pip install "loom-harness[anthropic]"   # + live Claude
pip install "loom-harness[mcp]"         # + MCP 网关
```

Python 3.10–3.13 · MIT · `import loom`

> Loom 缩小 agent 的爆炸半径、让它的行为可审查 —— 但它**不**保证模型不会作恶。见[威胁模型](docs/threat-model.md)。
