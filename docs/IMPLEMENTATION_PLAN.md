# Coding Agent 详细实施方案（第一版）

最后更新：2026-08-27

## 1. 项目定位

暂定名称：`MiniCoder`

一句话介绍：一个从零实现的、安全边界清楚、能自主修改代码并在验证失败后继续修复的命令行编程智能体。

目标不是复刻完整 Claude Code，而是完成一个结构完整、行为可信、两分钟内能演示、每行代码都能解释的中小型工程产品。正式代码目标为 2200 到 2800 行，硬上限约 3000 行；测试和文档不计入这个目标。

## 2. 完成标准

满足以下条件才算项目完成：

- 用户输入一个真实编程任务后，Agent 能连续多轮调用模型和本地工具。
- Agent 至少能浏览项目、读取文件、搜索文本、创建或精确修改文件、运行测试命令。
- 所有文件工具都不能越过指定 workspace。
- 命令具有工作目录、超时、输出截断和基础危险命令保护。
- API 或工具出错后不会直接崩溃，而会产生可理解的错误结果或有限次数重试。
- Agent 有最大步数和明确终止条件，不会无限循环。
- 修改代码后，Agent 会先验证再完成；测试失败时能继续尝试修复。
- 长对话不会无限增长，并能保留原始任务和关键事实。
- 核心模块有单元测试，完整循环有 Fake LLM 集成测试，真实模型有冒烟测试。
- README、演示脚本和面试题库与最终实现一致。

## 3. 用户看到的使用方式

预期命令：

```bash
python -m minicoder --workspace ./demo_project
```

启动后输入：

```text
请给这个待办事项 CLI 增加优先级功能，补充测试并确保现有测试通过。
```

终端只展示可审计信息：

```text
[MODEL] 请求读取 pyproject.toml
[TOOL] read_file 成功
[MODEL] 请求搜索 Todo 数据结构
[TOOL] search_text 找到 4 处
[MODEL] 请求修改 src/todo.py
[TOOL] replace_text 成功
[MODEL] 请求运行 pytest -q
[TOOL] 命令失败，1 个测试未通过
[MODEL] 根据错误继续修改
[TOOL] pytest -q 成功
[DONE] 功能完成并通过 12 个测试
```

不展示模型隐藏思维过程，只展示动作、结果和简洁说明。

## 4. 总体架构与数据流

整体采用六边形架构。Agent 核心只依赖 Port 接口，不直接依赖 DeepSeek SDK、具体操作系统或终端实现。详细模式选择见 `ARCHITECTURE_PATTERN_RESEARCH.md`。

```text
用户任务
   |
   v
CliAdapter -> AgentEngine -> ContextStrategy -> ModelPort
                  ^                              |
                  |                              v
                  +---- ToolResult <- ToolPipeline <- ToolCall
                                         |
                                Registry / Policies
                                         |
                          FileAdapters / ProcessAdapter

AgentEngine -> EventBus -> ConsoleSink / JsonlSink / TestSink
```

核心原则：模型决定“想调用哪个工具”，宿主程序决定“该调用是否合法、如何在本地执行、执行结果如何返回”。

## 5. 目录结构

```text
src/minicoder/
  __init__.py
  __main__.py
  bootstrap.py            # ApplicationFactory 和手工依赖注入
  config.py               # 环境变量、CLI 参数、跨平台默认值
  domain/
    models.py             # Message、ToolCall、ToolResult 等值对象
    events.py             # 事件类型
    state.py              # AgentPhase 和运行状态
    errors.py             # 内部错误分类
  application/
    ports.py              # ModelPort、ToolPort、EventSinkPort、ProcessPort
    agent_engine.py       # 核心循环和状态转换
    context.py            # 上下文预算与结构化摘要策略
    completion.py         # 验证后完成策略
    retry.py              # API 瞬时错误重试策略
    event_bus.py          # Observer 发布订阅
  tools/
    base.py               # Tool 接口
    registry.py           # Registry / Command 分发
    pipeline.py           # 责任链中间件
    validation.py         # JSON Schema 参数校验
    files.py              # 本地文件工具
    process.py            # 跨平台命令工具
    safety.py             # 路径与命令策略
  adapters/
    deepseek_chat.py      # Adapter/Gateway
    process_posix.py      # macOS/Linux Process Adapter
    process_windows.py    # Windows Process Adapter
    console.py            # CLI 和 Console Event Sink
    jsonl_trace.py        # JSONL Event Sink
tests/
  unit/
  integration/
examples/
  demo_project/         # 视频使用的稳定小项目
docs/
  PROJECT_MEMORY.md
  IMPLEMENTATION_PLAN.md
  INTERVIEW_QUESTION_BANK.md
```

预计正式代码 2200 到 2800 行，测试 1200 到 1800 行。若正式代码超过约 3000 行，必须检查是否出现为了展示模式而制造的无效抽象。

## 6. 核心模块设计

### 6.1 `Config`

负责把环境变量和命令行参数转换成明确配置：

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`，默认 `https://api.deepseek.com`
- `DEEPSEEK_MODEL`，默认 `deepseek-v4-pro`
- `workspace`
- `max_steps`，默认 20
- `command_timeout`，默认 30 秒
- `max_tool_output_chars`，默认 12000 字符
- `context_budget_chars`，先用字符近似控制，后续根据模型再决定是否引入 tokenizer

启动时统一校验，避免运行到一半才发现缺配置。

### 6.2 `ModelPort` 与 `DeepSeekChatAdapter`

只做四件事：

1. `ModelPort` 定义核心真正需要的接口。
2. Adapter 接收内部消息和工具 schema。
3. Adapter 调用普通 OpenAI 兼容 Chat Completions/tool calling API。
4. Adapter 把供应商返回值转换成项目内部的 `AssistantTurn`。
5. Adapter 把网络、限流、认证和格式错误转换为项目自己的异常类型。

它不负责执行工具、不负责管理完整会话，也不包含 Agent 决策。

### 6.3 `AgentEngine` 与显式状态机

显式状态：

```text
READY -> CALL_MODEL -> EXECUTE_TOOLS -> CALL_MODEL
                    \-> COMPLETE
                    \-> FAILED
```

每轮流程：

1. 检查是否超过最大步数。
2. 由 `ContextManager` 生成本轮消息。
3. 调用模型。
4. 保存模型回复。
5. 如果存在 tool calls，逐个交给 `ToolRegistry` 执行并保存结果。
6. 如果不存在 tool calls，检查“完成门槛”。
7. 满足门槛则结束；不满足则追加一条可见反馈，让模型继续验证。

终止条件：

- 模型返回最终文本，且满足完成门槛。
- 达到最大步数。
- 用户中断。
- 连续 API 错误超过重试上限。
- 出现宿主程序无法安全恢复的内部错误。

### 6.4 `ToolRegistry`、Command 与执行流水线

Registry 保存：工具名称、说明、JSON 参数 schema 和本地处理器。模型 `ToolCall` 被视为 Command 数据，先经过责任链流水线，再交给处理器。

职责：

- 向模型提供工具 schema。
- 根据名称查找工具。
- 解析 JSON 参数。
- 捕获工具异常并统一返回 `ToolResult`。
- 拒绝未知工具和非法参数。
- 依次应用参数验证、安全策略、执行、输出标准化和事件发布中间件。

统一返回结构建议：

```python
@dataclass
class ToolResult:
    ok: bool
    content: str
    error_code: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
```

### 6.5 文件工具

第一版只保留六个容易理解的工具：

| 工具 | 用途 | 关键边界 |
|---|---|---|
| `list_files` | 查看目录树 | 深度和条目数有限制，忽略 `.git` 等目录 |
| `read_file` | 分段读取文本 | 限制字符数，二进制文件返回明确错误 |
| `search_text` | 搜索关键词 | 限制目录、匹配数量和输出长度 |
| `create_file` | 创建新文件 | 默认不覆盖已有文件 |
| `replace_text` | 唯一匹配后精确替换 | 匹配 0 次或多次都拒绝，防止误改 |
| `delete_file` | 删除单个文件 | 第一版可选；若保留必须要求显式确认且禁止目录递归删除 |

默认不提供“任意内容覆盖已有文件”的工具。修改已有文件优先使用 `replace_text`，这样 diff 小、错误容易定位、面试也容易解释。

### 6.6 跨平台 `ProcessTool`

使用 `subprocess` 在 workspace 中同步执行命令：

- 固定 `cwd=workspace`。
- 设置超时。
- 同时捕获 stdout 和 stderr。
- 返回 exit code。
- 输出过长时保留开头和结尾，并标记已截断。
- 基础拦截明显危险程序和参数，例如提权、关机、磁盘格式化、递归删除宽泛路径。
- 第一版不支持交互式命令，不支持长期后台进程。
- 工具参数使用 `argv: list[str]`，默认 `shell=False`，避免依赖 Bash 或 PowerShell 的字符串解析。
- Python 命令通过 `sys.executable` 归一化。
- POSIX 与 Windows 在需要终止进程树时使用不同 Process Adapter，由 Factory 根据平台装配。

不支持管道、重定向和 `&&` 是有意取舍：Agent 可以发起多个顺序工具调用；这样同时提高安全性和跨平台一致性。

### 6.7 `SafetyPolicy`

文件路径流程：

1. 将模型提供的相对路径拼到 workspace。
2. 调用 `resolve()` 消除 `..` 并解析符号链接。
3. 用 `relative_to(workspace)` 验证最终路径仍在 workspace 内。
4. 不满足则返回 `PATH_OUTSIDE_WORKSPACE`，不访问文件。

需要测试普通路径、`../` 穿越、绝对路径和指向外部的符号链接。

### 6.8 `ContextManager`

上下文分成四层：

1. 永久层：系统规则和原始用户任务，永不删除。
2. 摘要层：较早历史的结构化摘要。
3. 最近层：最近若干轮模型回复和工具结果，保留原文。
4. 工作状态层：已修改文件、最近验证结果、剩余步数。

压缩分两步：

- 工具执行时就限制单条输出，防止一次测试日志占满上下文。
- 总历史超过预算时，将较早消息总结成“已完成动作、关键发现、未解决问题、文件变化、验证状态”，同时保留最近消息。

第一版用字符数近似预算，理由是供应商可替换且实现易懂。必须在 README 中说明它不是精确 token 计算。

### 6.9 验证后完成

Agent 维护：

- `last_mutation_step`
- `last_successful_verification_step`
- `modified_files`

当模型给出最终答案时：

- 如果没有修改文件，可以正常结束。
- 如果修改过文件，但修改后从未成功执行验证命令，则拒绝立即结束，并提醒模型运行测试、编译或至少执行语法检查。
- 如果最近一次验证失败，则继续把错误反馈给模型。
- 达到最大步数仍失败，则以“未完全完成”结束，并准确报告剩余问题。

这里的关键不是保证模型永远正确，而是让“完成”具有可检查证据。

### 6.10 事件记录

每个动作产生事件，例如：

- `task_started`
- `model_requested`
- `tool_called`
- `tool_finished`
- `verification_passed`
- `context_compacted`
- `task_completed`
- `task_failed`

默认输出简洁终端日志；开启 `--trace` 时写入本地 JSONL，便于调试和解释完整运行过程。trace 不保存 API Key。

## 7. 错误处理矩阵

| 错误 | 行为 |
|---|---|
| API Key 缺失 | 启动即失败，给出配置提示 |
| 401/403 | 不盲目重试，提示认证错误 |
| 429/5xx/网络抖动 | 指数退避，最多重试 2 到 3 次 |
| 模型返回未知工具 | 作为工具错误返回模型，让其重新选择 |
| tool arguments 不是合法 JSON | 返回 `INVALID_ARGUMENTS` |
| 路径越界 | 返回 `PATH_OUTSIDE_WORKSPACE` |
| 文件不存在/匹配不唯一 | 返回明确错误，不产生部分修改 |
| 命令超时 | 终止子进程并返回已有输出 |
| 输出过长 | 截断并明确标记 |
| 达到最大步数 | 安全结束并报告已完成和未完成内容 |

## 8. 测试方案

### 单元测试

- 路径安全：正常、`../`、绝对路径、符号链接逃逸。
- 文件工具：创建、读取分页、唯一替换、零匹配、多匹配、二进制文件。
- 命令工具：成功、非零退出、stderr、超时、输出截断、危险命令。
- Registry：正常分发、未知工具、非法 JSON、缺少参数。
- Context：预算内不压缩、超预算压缩、永久信息不丢失。

### Fake LLM 集成测试

Fake Client 依次返回预先设计的 tool calls，用来证明：

- Agent 能执行多步调用并正确追加 tool result。
- 修改后未验证时不会直接结束。
- 测试失败后能进入下一轮修复。
- 达到最大步数会终止。

### 真实 API 冒烟测试

只验证一次短任务：读取一个小文件、修改一处、运行测试并结束。它不替代可重复测试，也不能包含在默认 CI 中。

## 9. 两分钟视频演示方案

建议任务：给一个小型 Python 待办 CLI 增加“优先级”字段、更新排序逻辑并补充测试。

视频需要清楚出现：

1. 输入任务。
2. Agent 浏览和读取项目。
3. Agent 精确修改多个文件。
4. 第一次测试出现一个可控失败。
5. Agent 根据错误继续修复。
6. 第二次测试通过。
7. 展示最终 diff 或实际运行结果。

控制在 90 到 110 秒，给开头项目介绍和结尾留时间。允许加速，但关键 tool call 和失败修复不能全部剪掉。

## 10. 6 天实施节奏

### 8 月 27 日：冻结范围

- 确认用户基础、模型、系统和 CLI 形态。
- 初始化 Git 和公开仓库。
- 写最小 README 骨架、依赖和配置示例。
- 明确演示任务。

### 8 月 28 日：跑通最小闭环

- 实现 Config、ModelClient、ToolResult、Registry。
- 实现 list/read/create/replace/run_command。
- Agent 能完成三到五步真实任务。
- 每完成一个小功能就产生清晰 Git commit。

### 8 月 29 日：安全和错误处理

- 实现 workspace 路径边界、命令超时、输出截断和风险拦截。
- 补单元测试。
- 练习解释 tool calling 与本地工具执行的区别。

### 8 月 30 日：亮点功能

- 实现上下文管理。
- 实现验证完成门槛和失败修复循环。
- 加入 Fake LLM 集成测试和事件 trace。

### 8 月 31 日：完整验收

- 真实 API 多次运行演示任务。
- 修复稳定性问题，不再增加大功能。
- 完成 README.txt 初稿和面试题答案。

### 9 月 1 日：视频和答辩

- 录制、剪辑并检查 2 分钟/200 MB 限制。
- 用新环境按 README 重装运行。
- 进行至少两轮模拟面试。

### 9 月 2 日：提交缓冲

- 只修严重问题，不做架构重写。
- 检查仓库公开、提交历史、密钥、README、视频、ZIP 名称和提交页面。
- 在截止前完成最终 push 和提交，之后不再向仓库推送。

## 11. 可能的降级方案

若工期不足，按以下顺序降级：

1. 删除可选 `delete_file`，不影响核心能力。
2. JSONL trace 仅保留终端事件，不影响 Agent 闭环。
3. 上下文摘要改为确定性的旧消息截断与关键状态保留，但 `ContextManager` 接口不变。

不能降级的内容：本地工具执行、Agent 循环、终止条件、错误处理、路径安全、验证演示、测试和面试理解。

## 12. 增量开发与同步学习方式

每个增量严格执行：

1. 先说明本次只解决什么问题，以及为什么现在解决。
2. 编写不超过约 150 到 300 行的生产代码和对应测试。
3. 运行测试并查看 diff。
4. 逐文件、逐类、逐关键参数解释。
5. 比较至少一个替代方案，并说明能否不用该技术。
6. 在问题库增加与本增量代码直接绑定的问题。
7. 用户完成复述或追问后，再进入下一增量。
8. 创建一个含义清楚的 Git commit；题库记录 commit hash。

建议增量顺序：

- I01：项目骨架、Config、领域值对象和 ApplicationFactory。
- I02：ModelPort、DeepSeekChatAdapter 和 FakeModelAdapter。
- I03：Tool Command、Registry 和 JSON Schema 参数校验。
- I04：文件工具和 workspace 路径安全。
- I05：跨平台 ProcessPort、POSIX/Windows Adapter 和命令策略。
- I06：AgentEngine 最小循环与显式状态机。
- I07：Observer EventBus、ConsoleSink 和 JSONL trace。
- I08：责任链工具流水线和统一错误模型。
- I09：上下文预算、确定性裁剪和摘要 Strategy。
- I10：验证完成 Policy、失败修复和最大步数。
- I11：端到端测试、三平台 CI 和演示项目。
- I12：README、视频脚本、密钥扫描和最终验收。

## 13. 已确认与剩余事项

- 已确认 Python、CLI、DeepSeek V4 Pro、较高每日投入和三平台兼容。
- GitHub 公开仓库尚未创建；题目要求公开仓库，所以仅本地 Git 不足。
- 视频演示任务尚待用户理解后冻结，但不阻碍 I01 启动。
