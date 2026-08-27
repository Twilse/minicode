# 软件架构与设计模式调研及选型

最后更新：2026-08-27

## 1. 调研原则

设计模式不是越多越好。模式必须从具体问题出发，并同时记录收益、代价和不采用的替代方案。微软架构中心也强调：应按问题和约束选择模式，而不是先选技术或模式再寻找使用场景。

主要资料：

- [Alistair Cockburn：六边形架构原始文章](https://alistair.cockburn.us/hexagonal-architecture)
- [Microsoft：Architecture Styles](https://learn.microsoft.com/en-us/azure/architecture/guide/architecture-styles/)
- [Microsoft：Cloud Design Patterns](https://learn.microsoft.com/en-us/azure/architecture/patterns/)
- [Microsoft：Observer Design Pattern](https://learn.microsoft.com/en-us/dotnet/standard/events/observer-design-pattern)
- [Martin Fowler：Patterns of Enterprise Application Architecture](https://martinfowler.com/eaaCatalog/)
- [Martin Fowler：Gateway](https://martinfowler.com/articles/gateway-pattern.html)

## 2. 常见整体架构风格

| 架构风格 | 核心思想 | 常见优点 | 主要代价 | 本项目结论 |
|---|---|---|---|---|
| 分层架构 | 表现层、业务层、数据/基础设施层逐层依赖 | 直观、学习成本低 | 容易出现业务逻辑跨层泄漏 | 部分采用目录分层，但不作为唯一架构原则 |
| 六边形架构 | 核心通过 Port 与外部交互，具体技术放在 Adapter | 核心可独立测试，模型、CLI、操作系统容易替换 | 接口和对象数量增加 | 采用，作为总体架构 |
| Clean/Onion | 依赖方向指向核心，基础设施依赖业务抽象 | 边界清晰、可测试 | 小项目可能过度抽象 | 借用依赖规则，不完整照搬全部层次 |
| 事件驱动 | 生产者发布事件，消费者独立订阅 | 解耦日志、UI、trace | 调试顺序和事件管理更复杂 | 仅在进程内用于可观测性，不引入消息队列 |
| 微内核/插件 | 稳定核心配合可插拔扩展 | 扩展性强 | 插件生命周期和兼容性复杂 | Tool Registry 具有轻量插件特征，不做完整插件系统 |
| 微服务 | 按业务能力拆为独立进程和部署单元 | 独立部署、故障隔离 | 网络、部署、数据一致性成本巨大 | 不采用；单机 CLI 没有这个需求 |
| CQRS/Event Sourcing | 写模型与读模型分离，以事件重建状态 | 审计和复杂业务能力强 | 心智、存储和一致性成本高 | 不采用；JSONL trace 不是 Event Sourcing |
| MVC/MVVM | 分离界面、状态和交互逻辑 | 适合图形或 Web UI | 本项目没有复杂视图 | 不采用；CLI 是一个入站 Adapter |

## 3. 选定的总体架构：六边形架构

内部核心不直接认识 DeepSeek SDK、终端颜色、文件系统实现或具体操作系统。

```text
                    入站 Adapter
              CLI / 测试驱动程序
                       |
                       v
              +------------------+
              | Application Core |
              | AgentEngine      |
              | Policies / State |
              +------------------+
                 ^      ^      ^
                 |Port  |Port  |Port
                 |      |      |
        DeepSeekAdapter |   EventSink Adapters
                        |
                 Local Tool Adapters
```

### Ports

- `ModelPort`：核心需要“获得一次模型回复”，但不知道 OpenAI SDK 类型。
- `ToolPort`：核心需要“获得 schema 并执行一次工具调用”。
- `EventSinkPort`：核心发布运行事件，但不知道是打印、写 JSONL 还是被测试收集。
- `ProcessPort`：命令工具需要运行子进程，但平台实现可以不同。

### Adapters

- `DeepSeekChatAdapter`：把内部消息转换为 OpenAI 兼容 Chat Completions 参数，再把响应转换回内部对象。
- `CliAdapter`：读取用户任务并展示事件。
- `LocalFileTool`、`LocalProcessTool`：执行真实本地操作。
- `FakeModelAdapter`、`MemoryEventSink`：测试替身。

## 4. 采用的代码级设计模式

### 4.1 Adapter / Gateway

问题：DeepSeek/OpenAI SDK 返回对象、异常和参数名不应渗透到 Agent 核心。

实现：`DeepSeekChatAdapter` 实现内部 `ModelPort`，负责协议翻译。它同时具有 Gateway 的作用：外部 API 只从这里进入系统。

不用会怎样：AgentEngine 会到处出现 `client.chat.completions.create`、SDK 异常和厂商字段，难以测试或切换供应商。

### 4.2 Strategy

问题：上下文压缩、重试、完成判断和命令风险策略都有多种合理实现。

实现：

- `ContextCompactionStrategy`
- `RetryStrategy`
- `CompletionPolicy`
- `CommandSafetyPolicy`

核心只依赖策略接口，由启动装配阶段选择实现。

不用会怎样：大量条件分支进入 AgentEngine，后续更换策略会修改主循环。

为什么不使用继承模板方法：这些算法需要独立组合，而不是共享一个固定的父类流程；组合比深继承更清楚。

### 4.3 Observer

问题：同一次 Agent 事件需要同时显示到终端、写入 JSONL，并被测试捕获，但核心不应直接调用三个对象。

实现：`EventBus.publish(event)` 通知零个或多个 `EventSink`。

不用会怎样：每增加一种输出都要修改 AgentEngine。

为什么不是分布式消息队列：事件只在单进程内使用，不需要持久化投递、消费确认或网络通信。

### 4.4 Command

问题：模型返回的是“以后要执行的动作描述”，需要统一表示、校验、执行和记录。

实现：内部 `ToolCall` 是命令数据；每个 Tool 是命令处理器，统一返回 `ToolResult`。

不用会怎样：模型协议、工具名称和具体函数调用会通过 `if/elif` 散落在主循环。

### 4.5 Registry + Factory

问题：模型需要获得所有工具 schema，运行时还要按名称找到处理器；同时不同操作系统和运行配置需要构造不同 Adapter。

实现：

- `ToolRegistry` 保存工具名到 Tool 的映射，并导出 schema。
- `ApplicationFactory` 根据 Config 创建 Model Adapter、Process Adapter、Policies、Event Sinks 和 AgentEngine。

不用 Factory 会怎样：CLI 入口会充满对象构造和平台判断，测试难以替换依赖。

为什么不使用 Singleton：全局单例会污染测试和隐藏依赖；所有对象通过显式构造参数传入。

### 4.6 Chain of Responsibility / Middleware Pipeline

问题：每个工具调用都要依次经过参数校验、安全策略、执行、输出限制、事件记录。把这些逻辑复制进每个工具会重复且容易遗漏。

执行链：

```text
ToolCall
 -> JSON Schema 参数校验
 -> 工作区/命令安全检查
 -> 工具执行
 -> 输出截断与标准化
 -> 事件发布
 -> ToolResult
```

每个中间件可继续调用下一个中间件，也可提前返回错误结果。

不用会怎样：每个文件工具都要自己处理日志、异常和截断，行为不一致。

### 4.7 显式有限状态机

问题：Agent 具有“调用模型、执行工具、等待验证、完成、失败”等状态，单靠布尔变量容易产生非法组合。

实现：使用 `AgentPhase` 枚举和显式转换检查，但不为每个状态创建一个类。

为什么不完整使用 GoF State 类模式：状态行为数量有限，类数量膨胀的成本高于收益；枚举加转换函数足以表达约束。

## 5. 三个最终技术亮点

### 亮点一：六边形 Agent 内核

组合 Adapter/Gateway、Factory、Strategy 和依赖倒置。真实 DeepSeek 与 Fake Model 走同一 Port，可在无网络测试中完整驱动 Agent。

### 亮点二：策略化安全工具流水线

组合 Command、Registry、责任链、JSON Schema、本地 workspace capability 和跨平台 Process Adapter。模型可以提出动作，但宿主逐层控制动作是否被执行。

### 亮点三：结构化记忆与证据驱动状态机

上下文采用确定性裁剪加结构化摘要；完成策略追踪修改和验证证据。模型过早结束或验证失败时，状态机继续驱动修复，并由 Observer 产生可审计 trace。

## 6. DeepSeek 官方接口核对

官方资料：

- [DeepSeek API 首页](https://api-docs.deepseek.com/)
- [Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/)
- [Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)
- [Responses API](https://api-docs.deepseek.com/api/create-response/)
- [更新记录](https://api-docs.deepseek.com/updates/)

确认结果：

- 模型参数是 `deepseek-v4-pro`，用户记忆正确。
- OpenAI 格式 base URL 是 `https://api.deepseek.com`。
- API Key 是平台生成的密钥值，不存在模型专属的“API Key 名称”。环境变量名由应用自行约定，本项目使用 `DEEPSEEK_API_KEY`。
- 另设 `DEEPSEEK_MODEL=deepseek-v4-pro` 和 `DEEPSEEK_BASE_URL=https://api.deepseek.com`。
- DeepSeek tool calling 只返回结构化调用，具体函数必须由客户端执行，符合题目要求。
- 官方搜索缓存与当前 Responses API 页面在 V4-Pro 支持状态上出现过冲突。为了降低临近截止日期使用新接口的风险，第一版采用长期明确支持 V4-Pro 的 Chat Completions；内部 `ModelPort` 允许以后添加 Responses Adapter。
- 不依赖 DeepSeek 的服务端 web search、代码执行或文件工具。

## 7. 跨平台设计

目标平台：macOS、Linux、Windows。

- 路径全部使用 `pathlib.Path`，不手工拼接 `/` 或 `\\`。
- Python 解释器使用 `sys.executable`，不假设命令名一定是 `python3`。
- `run_command` 接收 `argv: list[str]`，默认 `shell=False`，不依赖 Bash、PowerShell 的管道和引号规则。
- 使用 `subprocess` 的跨平台基础参数；如需终止进程树，POSIX 和 Windows 由不同 `ProcessAdapter` 实现。
- 文本文件明确使用 UTF-8，并定义解码失败行为。
- 临时目录使用 `tempfile`，不写死 `/tmp`。
- 颜色输出检测 TTY，也提供 `--no-color`。
- 使用 GitHub Actions 的 Ubuntu、macOS、Windows matrix 运行测试，作为兼容性证据。

每实现一个涉及平台差异的增量，必须在讲解和面试题中明确标出差异与测试证据。

## 8. 明确不采用的模式

- Singleton：隐藏依赖，导致测试相互污染。
- 大量抽象工厂：当前只有少量 Adapter，一个 `ApplicationFactory` 足够。
- Repository：项目没有领域实体数据库；会话 JSONL 只是基础设施存储。
- MVC/MVVM：没有复杂 GUI。
- 微服务：单机 CLI 无独立部署需求。
- CQRS/Event Sourcing：状态规模小，trace 只用于审计，不用于重放恢复业务状态。
- 完整 DDD：Agent 的领域对象会认真命名，但不引入聚合、领域服务等整套术语。
- 完整 GoF State 类层次：枚举状态机更符合项目规模。

这些“不采用”同样是面试设计决策的一部分。
