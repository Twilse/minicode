# MiniCoder

MiniCoder 是本地命令行编程 Agent。模型负责决策，文件、搜索、命令和验证由本机在指定工作区执行，支持具备 Tool Calling 的 OpenAI-compatible Chat Completions 服务。

## 新机器运行

需要 Python 3.11 以上版本。`pyproject.toml` 能安装项目依赖，但不能安装 Python 本身。

HTTPS: `https://github.com/Twilse/minicode.git`

SSH: `git@github.com:Twilse/minicode.git`

```bash
git clone https://github.com/Twilse/minicode.git
cd minicode
python3 -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

export MINICODER_API_KEY="你的Key"
export MINICODER_BASE_URL="服务商API地址"
export MINICODER_MODEL="模型ID"

python -m minicoder --check-config --workspace .

cd /path/to/project
minicoder
```

`minicoder` 默认使用当前目录；激活虚拟环境后，进入目标项目直接运行即可。一次性任务可执行 `minicoder --workspace . "检查并测试项目"`。交互模式输入 `/exit` 或 `/quit` 退出；测试使用 `python -m pytest`。不要提交 API Key。

## 记忆机制

内存保存当前进程完整的 User、Assistant 和 Tool 消息。Session Archive 将任务、模型请求与回复、工具结果和终态追加到 `~/.minicoder/sessions/`；同工作区重启会恢复最近完整或中断的历史。它可能含源码和秘密，不是脱敏 Trace。

长期事实写入 `~/.minicoder/memory/`。开启时每轮结束进行一次无工具判断，模型只能返回 `append` 或 `none`；仅稳定、重要、有证据且不重复的事实可追加，默认不记，判断失败也不创建。全部有效记忆随以后请求放入 System。

每次请求由系统指令、全部长期记忆、预算内历史和本轮输入组成；Archive 恢复内容只作为历史 Messages，不回放 System，避免重复。

上下文未超限时发送完整历史；超限后总结较早完整协议组，当前输入和最新工具结果保持原文。首次压到预算约 70%，以后总结“旧摘要＋新增旧记录”；原文仍留在 Archive，不重复送入模型。

## 状态机、循环与计划

主循环为：输入 → 无工具规划 → 模型与工具循环 → 验证 → 完成。`AgentStateMachine` 限制各阶段迁移，计划确认前不能执行工具。

Planning 默认开启。首次请求使用 `tools=()`，只能生成 1～5 项计划；解析后 `PlanProgress` 启动第 1 项。执行请求增加 `finish_plan_step(step, summary)`，Schema 只允许当前编号；宿主再检查单独调用、完整参数、编号和 600 字符内的非空摘要。错误返回 ToolResult 且不推进，合法调用只能从 N 到 N+1。

普通工具不推进计划，程序也不猜测其所属步骤。模型返回 ToolCall 就执行并继续；无 ToolCall 但计划或验证未通过则加入纠正消息。最终回答被接受才进入 `COMPLETE`；达到调用上限、不可恢复错误、验证无法支持或用户中断时进入 `FAILED` 并终止。

## 工具调用

`ToolRegistry` 注册八个工具：`list_files、read_file、search_text、create_file、write_file、replace_text、run_command、read_tool_output`，冻结定义并校验 JSON Schema，再执行 Tool 并关联 call ID 返回结果。路径不能越界；argv 命令拒绝提权、递归删除和破坏性 Git。长输出保存为 artifact，模型先看预览再分段读取。

## 模型输出解析

Adapter 检查模型的 choice、正文、ToolCall 和 reasoning，转换为 `AssistantTurn` 并清除内部标记。Engine 分流后，计划由 `PlanProgress` 解析，工具参数经 JSON 和 Schema 校验，无工具正文通过门禁才成为最终回答。`finish_plan_step` 独立解析；摘要须非空，长期记忆须为严格 JSON；非法输出不会执行。

## 架构与设计模式

```text
用户 → CLI（驱动适配器）
          ↓
AgentEngine + Domain（应用核心）
          ↓ 只依赖 Protocol 端口
ModelPort / ProcessPort / EventSinkPort / ProjectMemoryPort / SessionArchivePort / ToolPort
          ↑
OpenAI 兼容模型 / POSIX或Windows / Console / JSONL / ToolRegistry
```

六边形架构中，`domain` 保存领域状态，`application` 编排 Agent 并定义 Ports，`adapters` 对接外部服务，`tools` 实现本地能力，`bootstrap.py` 统一装配。

Factory 创建对象图；Adapter 隔离 SDK 和系统；Strategy 替换重试、压缩与摘要算法；Observer 向 Console/Trace 发布事件；Registry/Command 分派工具；依赖注入支持 Fake 测试。

## 自定义验证

内置规则未覆盖时，在工作区创建并人工检查 `.minicoder.toml`：

```toml
[verification]
commands = [["zig", "build", "test"], ["python", "scripts/verify.py"]]
```

每项按完整 argv 精确匹配，最多 20 项；修改后重启加载。模型声明 `purpose="verification"` 不能自行授权其他命令。

## 错误与重试

连接失败、限流和服务端 5xx 重试两次，等待 0.5 秒和 1 秒，且不增加模型步骤。无效 Key、权限、其他 4xx 和协议错误不重试。工具或命令错误转成 ToolResult 交给模型纠正；摘要失败使用后备，记忆或事件失败只告警。
