# 增量代码面试问题库

最后更新：2026-08-27

## 1. 核心规则

本题库不预先堆放与代码无关的泛问题。只有以下内容已经存在时，才正式增加对应问题：

- 已经写入并保留的需求或架构决策；或者
- 已经增加的生产代码和测试。

每次代码增量完成后，问题必须记录：

- 增量编号和 Git commit；
- 对应生产文件和具体类/函数；
- 对应测试；
- 问题、基础答案要点和追问方向；
- 用户当前掌握等级。

若后续代码删除或设计变化，相关问题必须同步修改或标记失效。

## 2. 掌握等级

- L0：不了解。
- L1：听解释能理解。
- L2：能脱稿用自己的话解释。
- L3：能结合代码和测试回答追问、定位故障并说明替代方案。

所有核心增量最终目标为 L3。

## 3. 问题记录格式

```text
问题 ID：Ixx-Qyy
增量/Commit：Ixx / <commit hash>
对应代码：path::Class.method
对应测试：path::test_name
问题：
基础答案要点：
深入追问：
替代方案：
能否不用：
跨平台影响：无 / macOS / Linux / Windows / 全部
当前等级：L0/L1/L2/L3
掌握证据：
```

## 4. P00：题目与方案阶段（已有内容）

这一组只对应 PDF 要求和当前架构方案，不冒充代码问题。

### P00-Q01：为什么普通模型 SDK 可以使用，Agent SDK 不可以？

- 对应内容：`PROJECT_MEMORY.md` 第 2、4 节。
- 基础答案要点：普通 SDK 只处理 API 协议；Agent SDK 往往已经提供循环、工具编排、上下文和错误处理，会替代题目要求自行实现的核心。
- 深入追问：怎样用代码结构证明没有偷用 Agent 框架？
- 当前等级：L1（已经做过一次概念解释，待用户复述）。

### P00-Q02：DeepSeek 的 tool calling 为什么符合题目要求？

- 对应内容：`ARCHITECTURE_PATTERN_RESEARCH.md` 第 6 节。
- 基础答案要点：模型只返回工具名和 JSON 参数；具体函数由本地程序验证和执行。DeepSeek 官方文档也明确说明模型不会执行函数。
- 深入追问：为什么即使使用 strict schema，客户端仍应验证权限和业务约束？
- 当前等级：L1。

### P00-Q03：模型 ID、API Key 和通用环境变量分别是什么？

- 对应内容：`PROJECT_MEMORY.md` 第 4 节。
- 基础答案要点：模型 ID 由供应商定义；API Key 是供应商生成的秘密凭证；`MINICODER_API_KEY`、`MINICODER_BASE_URL`、`MINICODER_MODEL` 是本项目供应商中立的配置入口。DeepSeek 只是本地示例。
- 深入追问：为什么不能把 API Key 写入 Config 默认值？
- 当前等级：L0，待首次 Config 增量讲解。

### P00-Q04：为什么总体选择六边形架构？

- 对应内容：`ARCHITECTURE_PATTERN_RESEARCH.md` 第 3 节。
- 基础答案要点：Agent 核心同时依赖模型 API、文件系统、进程和终端；通过 Port/Adapter 隔离外部技术，可使用 Fake Adapter 测试核心，也能替换模型或平台实现。
- 深入追问：如果只有一个模型供应商，Adapter 是否仍有价值？
- 替代方案：普通分层架构；直接在 AgentLoop 调 SDK。
- 当前等级：L0，待 I01/I02 结合代码讲解。

### P00-Q05：为什么不能把所有设计模式都用一遍？

- 对应内容：`ARCHITECTURE_PATTERN_RESEARCH.md` 第 1、8 节。
- 基础答案要点：模式会引入接口、对象和间接层；只有收益大于认知与维护成本时才采用。模式服务于问题，不是项目目标。
- 深入追问：本项目为什么不用 Singleton、微服务、完整 State 类模式？
- 当前等级：L0。

### P00-Q06：为什么第一版选择 Chat Completions？

- 对应内容：`ARCHITECTURE_PATTERN_RESEARCH.md` 第 6 节。
- 基础答案要点：Chat Completions 对 V4-Pro 的官方支持时间更长且文档一致；Responses API 在近期更新期间出现过搜索缓存和当前页面不一致。ModelPort 隔离了选择，后续可添加 Responses Adapter。
- 深入追问：如果正式开发前实测 Responses 更稳定，如何切换而不修改 AgentEngine？
- 当前等级：L0，待 I02 讲解。

### P00-Q07：为什么仅使用本地 Git 不满足提交要求？

- 对应内容：PDF 和 `PROJECT_MEMORY.md` 第 2、9 节。
- 基础答案要点：Git 是版本控制系统，本地仓库只能自己访问；题目明确要求公开 GitHub/Gitee 仓库，评委需要通过远程地址查看代码和提交历史。
- 深入追问：GitHub 与 Git 的关系是什么？
- 当前等级：L1。

## 5. 代码增量索引

| 增量 | 内容 | Commit | 问题数 | 当前等级 |
|---|---|---|---:|---:|
| I01 | 项目骨架、Config、领域值对象、Factory | `ee057c1`、`f5cafa1` | 13 | L1 |
| I02 | ModelPort、兼容模型/Fake Adapter | 未开始 | 0 | L0 |
| I03 | Tool Command、Registry、参数校验 | 未开始 | 0 | L0 |
| I04 | 文件工具与路径安全 | 未开始 | 0 | L0 |
| I05 | 跨平台进程工具与命令策略 | 未开始 | 0 | L0 |
| I06 | AgentEngine 与状态机 | 未开始 | 0 | L0 |
| I07 | Observer EventBus 与 trace | 未开始 | 0 | L0 |
| I08 | 工具责任链与错误模型 | 未开始 | 0 | L0 |
| I09 | 上下文压缩 Strategy | 未开始 | 0 | L0 |
| I10 | 完成 Policy 与失败修复 | 未开始 | 0 | L0 |
| I11 | E2E、三平台 CI、演示项目 | 未开始 | 0 | L0 |
| I12 | README、视频和最终安全检查 | 未开始 | 0 | L0 |

## 6. I01：项目骨架、配置、领域值对象和 Factory

### I01-Q01：为什么采用 `src/` 项目布局？

- 对应代码：`pyproject.toml`、`src/minicoder/__init__.py`。
- 对应测试：测试通过已安装的 `minicoder` 包导入代码。
- 基础答案要点：源码与仓库根目录分开，测试更接近用户安装后的导入方式，减少“只因当前目录在 `sys.path` 中才成功”的假象。
- 替代方案：直接使用根目录包；小脚本可用，但工程边界较弱。
- 能否不用：可以，但会降低打包和测试一致性。
- 当前等级：L1。

### I01-Q02：为什么 `AppConfig.from_environment` 接收一个 `Mapping`？

- 对应代码：`src/minicoder/config.py::AppConfig.from_environment`。
- 对应测试：`tests/unit/test_config.py::test_from_environment_uses_safe_defaults`、`test_from_environment_accepts_explicit_overrides`。
- 基础答案要点：生产环境默认读取 `os.environ`；测试可传普通字典，不修改全局进程环境，也不产生测试相互影响。
- 替代方案：函数内部始终直接读取 `os.environ`，再用 monkeypatch 测试。
- 能否不用：可以，但依赖会更隐蔽，测试更依赖全局状态。
- 当前等级：L1。

### I01-Q03：为什么配置要在启动阶段统一校验？

- 对应代码：`src/minicoder/config.py::_validated_base_url`、`_positive_int`、`_positive_float`。
- 对应测试：`tests/unit/test_config.py::test_from_environment_rejects_invalid_values`。
- 基础答案要点：尽早失败，错误位置清楚；进入 Agent 循环后可以假设步数、超时和 URL 已合法，减少每个调用点的重复判断。
- 深入追问：为什么 URL 同时允许 `http` 和 `https`？本地兼容网关可能使用 HTTP，真实公网应优先 HTTPS。
- 能否不用：不能完全不校验；可以延迟校验，但错误会更难定位。
- 当前等级：L1。

### I01-Q04：程序怎样避免在日志中泄露 API Key？

- 对应代码：`src/minicoder/config.py::AppConfig.__repr__`、`src/minicoder/cli.py::_configuration_summary`、`.gitignore`。
- 对应测试：`tests/unit/test_config.py::test_config_repr_does_not_reveal_api_key`、`tests/unit/test_cli.py::test_check_config_prints_safe_summary`。
- 基础答案要点：Key 从环境读取；`.env` 不入库；对象 `repr` 固定显示 `<hidden>`；CLI 只显示“已配置”，不显示前后缀。
- 深入追问：为什么只依靠 `.gitignore` 不够？因为已提交或主动打印的内容不会被它挽救。
- 当前等级：L1。

### I01-Q05：为什么领域对象使用 `frozen=True, slots=True`？

- 对应代码：`src/minicoder/domain/models.py` 中的四个 dataclass。
- 对应测试：`tests/unit/test_domain_models.py`。
- 基础答案要点：`frozen` 防止对象创建后字段被重新赋值，降低消息历史被意外修改的风险；`slots` 固定属性集合、阻止随意新增属性并减少对象开销。
- 替代方案：普通可变 dataclass、NamedTuple、Pydantic model。
- 能否不用：可以，但需要接受更弱的不变量保护；Pydantic 功能更多但依赖和隐藏行为也更多。
- 当前等级：L1。

### I01-Q06：`__post_init__` 在这里解决什么问题？

- 对应代码：`src/minicoder/domain/models.py::ToolCall.__post_init__`、`Message.__post_init__`、`ToolResult.__post_init__`。
- 对应测试：`test_tool_call_rejects_empty_protocol_fields`、`test_message_rejects_tool_call_on_user_role`、`test_failed_tool_result_requires_error_code`。
- 基础答案要点：dataclass 生成构造函数后立即检查跨字段不变量，例如只有 assistant 消息能带 tool calls、失败结果必须有 error code。
- 深入追问：类型提示为什么不能替代这些检查？类型提示通常不在运行时自动验证，而且无法表达所有跨字段关系。
- 当前等级：L1。

### I01-Q07：`tool_call_id` 为什么必须在工具结果中保留？

- 对应代码：`src/minicoder/domain/models.py::ToolCall`、`ToolResult.as_message`。
- 对应测试：`tests/unit/test_domain_models.py::test_tool_result_converts_to_correlated_tool_message`。
- 基础答案要点：一次模型回复可能包含多个工具调用，返回结果必须通过 ID 与原调用对应；只记录工具名在重复调用同一工具时会产生歧义。
- 深入追问：工具执行成功但回传错误 ID 时会怎样？模型 API 可能拒绝消息，或上下文关联错误。
- 当前等级：L1。

### I01-Q08：为什么 `ToolResult.metadata` 要复制后包成 `MappingProxyType`？

- 对应代码：`src/minicoder/domain/models.py::ToolResult.__post_init__`。
- 对应测试：`tests/unit/test_domain_models.py::test_tool_result_copies_metadata_to_preserve_immutability`。
- 基础答案要点：即使 dataclass 是 frozen，内部字典仍可被修改；先复制可隔离调用方原字典，代理对象阻止通过结果对象修改映射。
- 深入追问：这是深度不可变吗？不是；嵌套的 list/dict 仍可能可变，当前 metadata 只放简单标量。
- 能否不用：可以返回普通 dict，但会破坏“运行历史不可被悄悄改写”的约束。
- 当前等级：L1。

### I01-Q09：`ApplicationFactory` 为什么叫 Composition Root？

- 对应代码：`src/minicoder/bootstrap.py::ApplicationFactory`。
- 对应测试：`tests/unit/test_bootstrap.py::test_factory_creates_validated_bootstrap_context`。
- 基础答案要点：对象创建和依赖装配集中在系统边界，业务对象只通过构造参数接收依赖。后续兼容模型 Adapter、Process Adapter 和 Policy 都由这里选择并连接。
- 替代方案：在 CLI 和各模块中随用随创建；或使用依赖注入框架。
- 能否不用 Factory：可以用一个普通 `build_application` 函数；关键是保留单一、显式的装配位置，而不是类名本身。
- 当前等级：L1。

### I01-Q10：跨平台识别为什么单独封装？

- 对应代码：`src/minicoder/platforms.py::detect_operating_system`。
- 对应测试：`tests/unit/test_platforms.py::test_detect_operating_system_maps_python_platforms`。
- 基础答案要点：把 Python 的 `darwin/linux/win32` 等外部字符串转换成稳定内部枚举；后续 Factory 根据枚举选择 POSIX 或 Windows Process Adapter。
- 深入追问：WSL 会识别成什么？通常 `sys.platform` 是 Linux，因此走 Linux/POSIX Adapter。
- 跨平台影响：macOS、Linux、Windows 全部。
- 当前等级：L1。

### I01-Q11：CLI 为什么返回退出码而不是到处调用 `sys.exit`？

- 对应代码：`src/minicoder/cli.py::main`。
- 对应测试：`tests/unit/test_cli.py::test_check_config_returns_two_for_user_configuration_error`。
- 基础答案要点：`main` 可在测试中直接调用；0 表示成功，2 表示用户配置错误；只有最外层 `__main__` 把返回值转换为进程退出。
- 深入追问：为什么 `stdout`、`stderr` 和 `environ` 也作为参数？这是显式依赖，测试可用内存对象替换真实全局资源。
- 当前等级：L1。

### I01-Q12：`python -m minicoder` 和 `minicoder` 命令如何到达同一个入口？

- 对应代码：`src/minicoder/__main__.py`、`pyproject.toml` 的 `[project.scripts]`。
- 对应测试：`tests/unit/test_cli.py`；手工冒烟测试 `python -m minicoder --check-config`。
- 基础答案要点：`python -m` 执行包内 `__main__.py`；安装时 console script 调用 `minicoder.cli:main`；两者最终复用同一个函数。
- 深入追问：为什么 `raise SystemExit(main())` 而不是只调用 `main()`？前者把返回值设置为真实进程退出码。
- 跨平台影响：两种入口均不依赖 Bash 或 PowerShell。
- 当前等级：L1。

### I01-Q13：为什么配置变量不能命名为 `DEEPSEEK_*`？

- 增量/Commit：I01 修正 / `f5cafa1`。
- 对应代码：`src/minicoder/config.py::AppConfig.from_environment`、`_required_text`、`.env.example`。
- 对应测试：`tests/unit/test_config.py::test_model_settings_are_provider_neutral`。
- 基础答案要点：DeepSeek 是开发者本地选择，不是产品架构约束。通用变量让相同代码通过 base URL、模型 ID 和凭证连接任意 OpenAI 兼容 tool calling 服务。
- 深入追问：为什么不再给 base URL 和模型设置 DeepSeek 默认值？默认值仍会形成隐式供应商绑定，也可能让客户误把请求发到错误端点。
- 替代方案：为每个供应商设计独立变量，或增加 profile 配置文件；前者扩展成本高，后者可在后续供应商原生协议增多时采用。
- 能否支持所有模型：不能自动支持任意协议；第一版支持 OpenAI 兼容 Chat Completions/tool calling，原生 Anthropic 等协议需要新增 Adapter，但 AgentEngine 不变。
- 当前等级：L1。

## 7. I02 以后追加内容的位置

每完成一个增量，在这里增加一个二级标题和至少三道代码绑定问题。例如：

```text
## I01：项目骨架与配置

### I01-Q01：为什么 Config 使用不可变 dataclass？
对应代码：src/minicoder/config.py::Config
对应测试：tests/unit/test_config.py::test_missing_api_key
...
```

问题数量不追求平均：关键模块可以有 8 到 12 道，简单模块可以有 3 到 5 道。

## 8. 每次增量结束前的质量门槛

- 代码和测试已经实际存在并运行。
- 问题引用的类、函数和测试名称真实存在。
- 至少包含一个“为什么选它而不是替代方案”的问题。
- 至少包含一个错误或边界条件问题。
- 涉及平台行为时，必须包含跨平台问题。
- 用户完成一次复述；掌握等级按证据更新，而不是凭感觉填写。
