# LW 快速部署与实机启动指南

本文只给出 LW 从开发验证到实机启动的标准成功路径。设计依据、参数含义、异常
处理和安全边界以[《LW 编译与部署使用说明》](LW_BUILD_DEPLOYMENT_CN.md)为准；
两份文档有歧义时，停止操作并遵循完整说明。

> [!CAUTION]
> 未完成 Sim2Sim、部署包离线验收和现场安全检查时，不得启动实机节点。机器人
> 必须可靠吊装或离地，运动范围隔离，现场人员掌握物理急停；正常节点、参数测算
> 程序和其他工具不得同时读写两个电机板串口。

## 1. 先选择执行路径

| 当前情况 | 必须执行 |
| --- | --- |
| 首次实机部署 | 本文全部步骤，包括参数测算闭环 |
| 更换主机、内核、CPU/调度方案、IMU、串口设备、控制频率、策略模型、推理运行时或关键配置 | 重新执行参数测算闭环 |
| 已有可追溯的测算、评审和最终部署记录，且上述环境均未变化 | 可跳过第 5 节，从第 6 节日常启动检查继续 |
| 报告缺失、分析失败、评审关系不清或无法确认环境未变化 | 禁止启动；按首次部署重新测算 |

如果测算候选导致 `policy/LW/base.yaml` 改动，最终提交会与原测算候选提交不同。
交付记录必须保留“测算候选提交 → 原始报告 → 人工评审 → 最终配置提交”的关系，
不能只保存一个无法追溯来源的数值。

## 2. 统一设置路径

开发机和部署机分别在自己的项目中设置 `RL_SAR_ROOT`。`SOURCE_COMMIT` 必须是
已经通过开发机 Sim2Sim 的完整 40 位提交哈希，不能使用 `latest` 或口头约定的
“最新版本”。

```bash
RL_SAR_ROOT=/home/lfr/rl_sar
cd "$RL_SAR_ROOT"

SOURCE_COMMIT=REPLACE_WITH_FULL_40_CHARACTER_GIT_HASH
SHORT_COMMIT="${SOURCE_COMMIT:0:12}"
DEPLOY_PREFIX="$RL_SAR_ROOT/build/lw_deployments/$SHORT_COMMIT"
LW_PROFILE_DIR="$RL_SAR_ROOT/build/lw_profiles/$SHORT_COMMIT"
```

`SOURCE_COMMIT` 的占位符必须替换后才能执行。部署输出目录必须不存在或为空；
不要覆盖旧部署目录，同一提交重建时使用带后缀的新目录。

## 3. 开发机：构建、Sim2Sim 和提交

```bash
source /opt/ros/humble/setup.bash
cd "$RL_SAR_ROOT"
git status --short
./build.sh
scripts/validate_lw_strict_build.sh
source "$RL_SAR_ROOT/install/setup.bash"
ros2 run rl_sar rl_sim_LW
```

上述 `./build.sh` 不带参数时构建整个 ROS 工作区，是正式 Sim2Sim 验证的
标准入口。开发期可用 `./build.sh rl_sar` 指定包及其依赖；`-c`/`--clean`
不带包名时清理整个工作区，`./build.sh --clean rl_sar` 则只清理该包及其反向
依赖的 isolated `build/`、`install/` 子目录。完整用法可运行
`./build.sh --help`。交付前必须使用上述完整工作区构建
完成标准 `rl_sim_LW` Sim2Sim 验证。严格构建默认并行度为 2；
资源受限或需要调整时可显式使用：

```bash
LW_STRICT_BUILD_JOBS=4 scripts/validate_lw_strict_build.sh
```

上述默认命令不创建 Sim2Sim Plot publisher、timer 或快照缓冲。需要观察
`/LW_joint_states` 时，显式启用 Plot（默认 100 Hz）：

```bash
ros2 run rl_sar rl_sim_LW --enable-plot
```

也可选择 1–200 Hz 的整数频率，例如常规观察使用 50 Hz：

```bash
ros2 run rl_sar rl_sim_LW --enable-plot --plot-rate-hz 50
```

`--plot-rate-hz` 必须与 `--enable-plot` 同时使用；缺失、非整数、越界或重复
冲突值会使 Sim2Sim 明确拒绝启动。

Plot 开关只发布 `/LW_joint_states`，不会自动保存数据。需要用 PlotJuggler
离线分析时，保持上述 Sim2Sim 运行，并在另一个终端中执行：

```bash
mkdir -p bags
bag_dir="bags/lw_sim_$(date +%Y%m%d-%H%M%S)"
ros2 bag record \
    --output "$bag_dir" \
    /LW_joint_states
```

完成采集后先在录包终端按 `Ctrl+C`，等待 rosbag2 正常生成 `metadata.yaml`。
随后检查 bag 并启动 PlotJuggler：

```bash
ros2 bag info "$bag_dir"
ros2 run plotjuggler plotjuggler
```

在 PlotJuggler 中选择 ROS 2 Bag 数据加载器并打开 `$bag_dir` 对应的 bag。

Sim2Sim 默认直接读取编译时仓库根目录下的 `policy/`。该绝对路径
会写入可执行文件；`build/` 和 `install/` 不是默认策略来源。如需改用
其他策略目录，使用 `--policy-root PATH`：

```bash
ros2 run rl_sar rl_sim_LW \
    --policy-root /absolute/path/to/policy
```

有效策略根只需包含四套 ONNX 策略。Sim2Sim 的每个关节始终使用
MuJoCo PD+前馈力矩。`--policy-root` 可与 Plot 参数组合。编译后如果
移动仓库，需重新构建或显式指定新策略根。

至少确认四个 ONNX 策略均为本次候选版本，腿式、轮式和两个形态转换都能完整
运行，没有加载错误、NaN、越界、持续发散、明显跳变或控制周期异常。关闭仿真后
提交所有获批的代码、配置、模型和描述资源：

```bash
git status --short
git show --stat --oneline HEAD
RELEASE_TAG=lw-release-20260816-01
git tag -a "$RELEASE_TAG" -m "LW部署验证"
git push origin "$RELEASE_TAG"
SOURCE_COMMIT=$(git rev-parse "${RELEASE_TAG}^{commit}")
echo "$SOURCE_COMMIT"
```

把简短的标签名（例如 `lw-release-20260816-01`）和 Sim2Sim 结果交给
部署机，无需手工传递完整哈希。每个部署候选版本使用新标签，不要移动或
复用已验证标签。提交或建立标签后若又修改任何发布资源，必须重新测试、
Sim2Sim、提交并创建新标签。

## 4. 部署机：生成并完成候选部署包首次离线验收

本节对每个新候选版本完整执行一次。发布提交或标签、模型、配置、ONNX Runtime
发生变化，或者生成了新的 `DEPLOY_PREFIX`，都属于新候选，不能沿用旧候选的
验收结果。

在部署机自己的项目中新开终端：

```bash
source /opt/ros/humble/setup.bash
cd "$RL_SAR_ROOT"
git status --short
RELEASE_TAG=lw-release-20260816-01
git fetch origin tag "$RELEASE_TAG"
SOURCE_COMMIT=$(git rev-parse "${RELEASE_TAG}^{commit}")
git cat-file -e "${SOURCE_COMMIT}^{commit}"
git show --stat --oneline "$SOURCE_COMMIT"

src/rl_sar/scripts/build_lw_deployment.sh \
    "$DEPLOY_PREFIX" \
    "$SOURCE_COMMIT"
```

`build_lw_deployment.sh` 的形式为
`<empty-output-prefix> [commit]`；脚本虽允许省略提交并使用 `HEAD`，但本可追溯部署
流程必须显式传入已验证的完整 `SOURCE_COMMIT`。

构建脚本会先确认项目内 ONNX Runtime 目录存在，再使用候选提交中的
`validate_inference_runtime.sh` 按当前机器架构检查其结构、共享库和 ELF 架构；
只有校验通过才会创建部署输出并开始编译，无需另行手动执行该校验。

构建脚本必须成功验收原部署前缀及其迁移副本。随后在未加载其他工作区的新终端
重新执行第 2 节变量块，再执行：

```bash
source /opt/ros/humble/setup.bash
source "$DEPLOY_PREFIX/setup.bash"
"$DEPLOY_PREFIX/lib/rl_sar/rl_real_LW" --verify-deployment-only

for package in serial fdilink_ahrs rl_sar; do
    test "$(realpath -m "$(ros2 pkg prefix "$package")")" = "$DEPLOY_PREFIX"
done
```

`--verify-deployment-only` 必须返回 `0`，三个包必须解析到当前
`DEPLOY_PREFIX`。任一检查失败都停止部署，不得直接修改部署目录、伪造来源
文件或从其他版本补拷二进制、模型和动态库。

上述新终端检查是候选部署在最终运行位置的首次完整离线验收。同一前缀保持在
该位置，部署文件及 ROS、操作系统动态库等运行环境均未变化时，日常启动不要求
重复整段本节；前缀被复制或移动、文件完整性存疑、运行环境或依赖变化，或者
无法确认验收记录仍适用时，必须重新执行本节。

## 5. 首次部署或环境变化：参数测算闭环

参数测算产生的是供人工评审的候选，不会自动修改 `base.yaml`，也不能替代
Sim2Sim、物理急停或 STM32 侧失能确认。
包装工具及 `collect-host`、`collect-hardware`、`analyze` 子命令都支持
`-h`/`--help`；帮助模式只显示参数后退出，不运行 profiler 或访问硬件。

### 5.1 无硬件 CPU 和调度测量

在目标部署机上执行。此阶段不会初始化 ROS、IMU、手柄、串口或执行器设备：

```bash
source /opt/ros/humble/setup.bash
source "$DEPLOY_PREFIX/setup.bash"

LW_PROFILER="$DEPLOY_PREFIX/lib/rl_sar/lw_config_profiler"
LW_PROFILE_TOOL="$DEPLOY_PREFIX/lib/rl_sar/profile_lw_runtime_config.py"
LW_POLICY_ROOT="$DEPLOY_PREFIX/share/rl_sar/deployment/LW/policy"
LW_PROFILE_DIR="$RL_SAR_ROOT/build/lw_profiles/$SHORT_COMMIT"

python3 "$LW_PROFILE_TOOL" collect-host \
    --profiler "$LW_PROFILER" \
    --policy-root "$LW_POLICY_ROOT" \
    --output-dir "$LW_PROFILE_DIR/host" \
    --duration-seconds 30 \
    --cpus allowed \
    --realtime-priorities 0
```

`--cpus allowed` 会测量当前进程允许的全部逻辑 CPU；也可使用
`--cpus=-1,0,1` 选择不绑定对照组和指定 CPU。`--realtime-priorities`
接受逗号分隔的非负整数；只有部署负责人已批准正数优先级时，才可加入
`--require-realtime` 要求该正数优先级的 `SCHED_FIFO` 应用失败即中止。
`--duration-seconds` 必须为正数，默认 30；本流程显式写出默认值以固定
可比较的采样时长。

输出目录必须不存在或为空。未经部署负责人批准，不要加入正数实时优先级。

### 5.2 吊装硬件观察

开始前逐项确认：

- [ ] 机器人可靠吊装，运动范围内无人和障碍物；
- [ ] 现场人员掌握物理急停；
- [ ] 正常 `rl_real_LW` 和完整 LW launch 已停止；
- [ ] 两个电机板串口没有其他读写者；
- [ ] 当前终端加载的是同一个 `DEPLOY_PREFIX`。

只单独启动 AHRS，不启动完整 LW launch：

```bash
ros2 launch fdilink_ahrs ahrs_driver.launch.py
```

当前 `ahrs_driver.launch.py` 没有项目自定义 launch 参数，因此这条命令不应
追加串口或话题覆盖值。

在另一个终端重新执行第 2 节变量块以及 5.1 开头的 `source`、`LW_PROFILER`、
`LW_PROFILE_TOOL`、`LW_POLICY_ROOT` 和 `LW_PROFILE_DIR` 赋值，再执行以下命令；
确认字符串必须逐字匹配：

```bash
python3 "$LW_PROFILE_TOOL" collect-hardware \
    --profiler "$LW_PROFILER" \
    --policy-root "$LW_POLICY_ROOT" \
    --output "$LW_PROFILE_DIR/hardware.json" \
    --duration-seconds 60 \
    --cpu -1 \
    --realtime-priority 0 \
    --confirmation I_CONFIRM_LW_IS_SUSPENDED_AND_MOTORS_MUST_REMAIN_DISABLED
```

`collect-hardware` 还支持以下可选设置：

| 参数 | 默认值 | 约束 |
| --- | --- | --- |
| `--right-port` | `/dev/ttyLegRight` | 右电机板串口 |
| `--left-port` | `/dev/ttyLegLeft` | 左电机板串口 |
| `--imu-topic` | `/imu` | 独立 AHRS 驱动的原始 IMU 话题 |
| `--ahrs-topic` | `/euler_angles` | 独立 AHRS 驱动的欧拉角话题 |
| `--require-realtime` | 关闭 | 只能与正数 `--realtime-priority` 同时使用 |

只有在稳定设备别名或话题配置已单独评审、且测量对象与最终部署一致时，
才能追加端口或话题覆盖参数。`--duration-seconds` 必须为正数，默认 60；
`--cpu` 必须是 `-1` 或非负整数，`--realtime-priority` 必须是非负整数。

执行 `collect-hardware` 的整个测算过程中，程序不会使能电机；唯一允许进入两个
电机板串口的命令是 `motors_disable=true`。它先完成双侧初始失能写入，再启动
独立的 5 ms 失能保活，策略只做 shadow inference，输出全部丢弃。命令结束后
停止独立 AHRS 驱动。

该证明只表示上层完整写入失能包且没有发送非失能命令，不表示 STM32 已经执行
失能，也不能证明电机物理上已经失能。整个阶段仍必须保持吊装、隔离和物理急停。

### 5.3 生成评审候选并返回开发闭环

四个上限必须由机械与控制安全负责人确定。先填写变量；`analyze` 会拒绝占位符、
空值、非有限值和非正数：

```bash
MAX_SAFE_SENSOR_TIMEOUT_MS=REPLACE_WITH_REVIEWED_VALUE
MAX_SAFE_TRUSTED_IMU_TIMEOUT_MS=REPLACE_WITH_REVIEWED_VALUE
MAX_SAFE_IMU_AHRS_PAIR_AGE_MS=REPLACE_WITH_REVIEWED_VALUE
MAX_SAFE_CONTROL_GAP_MS=REPLACE_WITH_REVIEWED_VALUE

python3 "$LW_PROFILE_TOOL" analyze \
    --base-yaml "$LW_POLICY_ROOT/LW/base.yaml" \
    --reports "$LW_PROFILE_DIR"/host/*.json "$LW_PROFILE_DIR/hardware.json" \
    --output "$LW_PROFILE_DIR/candidate-review.json" \
    --max-safe-sensor-timeout-ms "$MAX_SAFE_SENSOR_TIMEOUT_MS" \
    --max-safe-trusted-imu-timeout-ms "$MAX_SAFE_TRUSTED_IMU_TIMEOUT_MS" \
    --max-safe-imu-ahrs-pair-age-ms "$MAX_SAFE_IMU_AHRS_PAIR_AGE_MS" \
    --max-safe-control-gap-ms "$MAX_SAFE_CONTROL_GAP_MS" \
    --minimum-hardware-samples 1000
```

`--reports` 接受一个或多个报告，但本流程要求同一部署的全部 host 报告和
一份 hardware 报告。四个 `--max-safe-*-ms` 在程序接口中可省略，但本实机
候选评审流程要求全部填写为由安全负责人确定的有限正数；参数解析和正数检查
由 `analyze` 自身执行。
`--minimum-hardware-samples` 必须是正整数，默认 1000；输出文件不得已经存在。

`candidate-review.json` 仅供评审。不得把它直接覆盖到当前部署包。需要采用候选时：

1. 在开发机源码树中修改 `policy/LW/base.yaml`；
2. 重新执行测试和 Sim2Sim；
3. 创建并记录新的最终提交；
4. 部署机从新提交生成新的 `DEPLOY_PREFIX`；
5. 对新前缀重新执行第 4 节的全部离线验收；
6. 保存测算候选、原始报告、评审结论和最终提交之间的追溯记录。

## 6. 日常实机启动前

以下轻量检查每次启动都执行。新候选首次启动前必须已经完成第 4 节的最终位置
完整验收；同一份未变化且已验收部署的日常重复启动无需重新构建，也无需手动
重复第 4 节的 `--verify-deployment-only` 和全部 `ldd` 检查。正常
`rl_real_LW` 启动仍会在打开电机板串口和访问硬件前自动校验部署清单及资源
哈希，不会因采用日常流程而绕过完整性保护。

- [ ] 当前 `DEPLOY_PREFIX` 与交付记录一致；
- [ ] 该前缀仍位于已验收的最终位置，部署文件和运行环境均未变化；
- [ ] 已确认该最终提交的测试和 Sim2Sim 记录；
- [ ] 首次部署/环境变化所需的测算与人工评审已经闭环；
- [ ] `/dev/ttyLegRight`、`/dev/ttyLegLeft`、`/dev/fdilink_ahrs` 正确且可访问；
- [ ] 没有其他进程读取 IMU 或两个电机板串口；
- [ ] 机器人可靠吊装或离地，运动范围隔离；
- [ ] 物理急停可用，现场监护人员已就位。

每次启动都在新终端重新加载基础 ROS 和已验收前缀，并做轻量包解析检查：

```bash
source /opt/ros/humble/setup.bash
source "$DEPLOY_PREFIX/setup.bash"

for package in serial fdilink_ahrs rl_sar; do
    test "$(realpath -m "$(ros2 pkg prefix "$package")")" = "$DEPLOY_PREFIX"
done

PYTHONDONTWRITEBYTECODE=1 ros2 launch rl_sar rl_real_LW.launch.py
```

真机 launch 只提供三个项目自定义参数：

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `enable_keyboard:=<boolean>` | `true` | 只接受 `true`/`false`；是否从控制终端读取真机键盘 |
| `enable_debug_publisher:=<boolean>` | `false` | 只接受 `true`/`false`；是否创建 `/LW_joint_states` 调试 publisher |
| `debug_publish_rate_hz:=<integer>` | `50` | 只接受 1–200；调试 publisher 频率，关闭 publisher 时仍校验参数但不创建资源 |

### 真机键盘和手柄速查

以下键位只在相应 FSM 状态允许时生效，不会跳过起身、趴下或形态切换过程。
键盘操作要求 `enable_keyboard:=true`，并且 launch 进程具有可访问的控制终端。

| 功能 | 键盘 | 手柄 | 生效条件或说明 |
| --- | --- | --- | --- |
| 腿式起身 | `0` | `A` | 在 Passive、轮式起身或 GetDown 状态请求腿式起身 |
| 进入腿式运动 | `1` | `RB` + 十字键上 | 仅在腿式起身完成后进入腿式 locomotion |
| 轮式起身 | `2` | `Y` | 在 Passive、腿式起身或 GetDown 状态请求轮式起身 |
| 进入轮式运动 | `3` | `RB` + 十字键下 | 仅在轮式起身完成后进入轮式 locomotion |
| 腿式切换为轮式 | `4` | `RB` + 十字键左 | 仅在腿式 locomotion 中启动转换 |
| 轮式切换为腿式 | `5` | `RB` + 十字键右 | 仅在轮式 locomotion 中启动转换 |
| 受控趴下 | `9` | `B` | 起身完成后、locomotion 或转换状态中请求 GetDown；Passive 中无动作 |
| 转入 Passive 阻尼 | `P` | `LB` + `X` | `Kp=0`、`Kd=5`，不等于电机失能或零执行器输出 |
| 切换导航模式标志 | `N` | `X` | 切换 `navigation_mode`；当前本地速度仍由摇杆输入 |

组合键应先按住肩键，再按面键或推动十字键。真机速度控制如下：

| 功能 | 键盘 | 手柄 |
| --- | --- | --- |
| 前向速度 `x` | 无 | 左摇杆纵向：向前为正、向后为负 |
| 横向速度 `y` | 无 | 左摇杆横向：向左为正、向右为负 |
| 偏航速度 `yaw` | 无 | 右摇杆横向：向左为正、向右为负 |
| 三轴速度清零 | 无 | 将已使用的摇杆回中 |

LW 键盘不提供速度控制，`W/S/A/D/Q/E` 和 `Space` 都不会修改速度；`Space`
不是停车键或急停。当前已提交策略的横向速度上限为 `0.0`，因此默认配置下左
摇杆横向不会产生 `y` 指令。`GetDown` 是受控运动，Passive 是阻尼状态，两者
都不能代替物理急停或电机失能。

当前真机按 `JOYSTICK_1` 逻辑布局解释 `/dev/input/js0`。不同手柄或驱动的按钮
编号可能不同，首次使用必须在机器人断电或可靠吊装时核对实际设备事件，不能只
根据手柄外壳标识判断按键。

无交互终端的 systemd、容器或后台运行必须显式关闭键盘：

```bash
PYTHONDONTWRITEBYTECODE=1 ros2 launch rl_sar rl_real_LW.launch.py enable_keyboard:=false
```

关闭键盘后必须事先准备独立可控恢复方式、可靠机械支撑和物理急停。只有受控
短时调试才启用 publisher：

```bash
PYTHONDONTWRITEBYTECODE=1 ros2 launch rl_sar rl_real_LW.launch.py \
  enable_debug_publisher:=true debug_publish_rate_hz:=50
```

调试 publisher 只发布新的控制源帧，发生争用或新帧覆盖时允许丢弃样本，不会
阻塞 200 Hz 控制循环。调试结束后应恢复默认关闭。三个参数可同时指定。

需要保存 `/LW_joint_states` 供 PlotJuggler 离线分析时，在同一控制机的第二个
终端中执行。将 `LW_BAG_ROOT` 替换为可写的绝对路径，且该路径必须位于受完整性
保护的 `DEPLOY_PREFIX` 之外：

```bash
LW_BAG_ROOT="/absolute/writable/path/outside/DEPLOY_PREFIX"
mkdir -p "$LW_BAG_ROOT"
bag_dir="$LW_BAG_ROOT/lw_real_$(date +%Y%m%d-%H%M%S)"
ros2 bag record \
  --output "$bag_dir" \
  /LW_joint_states
```

正常采集结束时先在录包终端按 `Ctrl+C`，等待 rosbag2 写入
`metadata.yaml`，然后检查 bag 并启动 PlotJuggler：

```bash
ros2 bag info "$bag_dir"
ros2 run plotjuggler plotjuggler
```

在 PlotJuggler 中选择 ROS 2 Bag 数据加载器并打开 `$bag_dir` 对应的 bag。录制会
在控制机上产生磁盘 I/O，必须先确认磁盘空间并限制采集时长。rosbag2 是外部
订阅者，录制失败或磁盘写满不会停止机器人；调试 publisher 也允许因争用丢弃
样本，因此 bag 不代表无损控制周期日志。机器人异常时必须优先执行物理急停和
失能，不能为保存 bag 延迟安全操作。调试完成后恢复默认关闭
`enable_debug_publisher`。

必须保留 `PYTHONDONTWRITEBYTECODE=1`，否则 Python 可能在清单约束的生产 launch
目录生成 `__pycache__`，完整性检查会安全拒绝启动。正常 launch 会访问真实
IMU、串口和电机；不能用它测试部署包是否完整。

## 7. 停止、失败和回滚

- 任一检查或启动步骤报错：停止操作，不绕过校验，不现场修改部署包。
- 机器人出现非预期动作：立即使用物理急停，并停止 launch。
- 回滚：选择上一个完整、已验收的 `DEPLOY_PREFIX`，重新执行第 4 节离线验收和
  第 6 节安全清单；不能只替换可执行文件、模型或 YAML。
- 依赖、运行时、串口规则、清单、ROS 工作区或动态库问题，查阅完整说明的
  [常见问题](LW_BUILD_DEPLOYMENT_CN.md#10-常见问题)。
