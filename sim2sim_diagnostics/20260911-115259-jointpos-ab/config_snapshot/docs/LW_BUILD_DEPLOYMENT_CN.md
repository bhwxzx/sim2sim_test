# LW 编译与部署使用说明

本文是 LW 编译、参数测算和实机部署的权威说明。只执行标准成功路径时可先阅读
[《LW 快速部署与实机启动指南》](LW_QUICK_START_CN.md)；遇到歧义、失败或需要
理解参数和安全边界时，以本文为准。

本文面向下面这种实际使用方式：

- **开发机**用于修改代码、模型和配置，并完成 Sim2Sim 仿真验证；
- **部署机**连接真实机器人，只进行正式部署验收和实机实验，同时也保留一份完整的 `rl_sar` 项目；
- 部署机在自己的 `rl_sar` 项目中，从开发机确认的 Git 提交生成一个固定的实机运行版本。

> [!CAUTION]
> 正常启动会连接传感器、串口和执行器。首次运行或更换代码、模型、配置、硬件环境后，必须先完成本文的离线验收。实机试运行时应将机器人可靠吊装或离地，保证急停可用，并安排人员现场监护。

## 1. 先理解三个目录

以下命令假设开发机和部署机的项目目录都是：

```text
/home/lfr/rl_sar
```

如果实际路径不同，只需修改命令中的 `RL_SAR_ROOT`。

部署机上会同时存在三类内容：

```text
/home/lfr/rl_sar/
├── src/、policy/、library/       # 完整项目：用于保存源码和外部推理库
└── build/lw_deployments/
    ├── <旧版本提交短哈希>/       # 以前验收通过的实机版本
    └── <当前提交短哈希>/         # 当前准备运行的实机版本
```

完整项目和部署版本并不冲突：

- 完整项目回答“有哪些源码可以修改和编译”；
- 部署版本回答“实机现在运行的究竟是哪一次提交、哪四个模型和哪些配置”。

部署版本不会取代完整项目。它只是把一次正式运行使用的可执行文件、模型和配置冻结下来，防止旧二进制、新源码、临时模型和不同版本配置混在一起。

## 2. 推荐流程总览

```text
开发机修改代码、配置、模型和其他发布资源
        ↓
开发机编译并完成 LW Sim2Sim 验证
        ↓
将全部获批发布资源提交到 Git
        ↓
创建并推送新的不可变发布标签
        ↓
部署机精确取得该标签并自动解析完整提交哈希
        ↓
部署机为该候选执行干净的正式构建和自动验收
        ↓
在最终运行位置完成该候选的首次离线验收
        ↓
完成机械和人员安全检查
        ↓
执行每次启动检查并进行实机实验
```

首次部署、软硬件环境变化或需要重新确定时序参数时，在已验收候选上执行
`collect-host`、吊装状态下的 `collect-hardware` 和 `analyze`。分析结果只供
人工评审；如果决定采用候选值，必须回到源码配置完成修改、测试、Sim2Sim、
新提交和新标签，再重新生成候选版本，不能直接修改现有部署目录。

推荐在部署机本地构建，而不是直接复制开发机编译出的二进制。这样可以使用部署机自己的 CPU 架构、ROS、编译环境和 `library/inference_runtime`，减少两台机器环境不一致造成的问题。

这里有三个职责不同的验收关口：

- 开发机的 **Sim2Sim 验证**检查策略和控制逻辑在仿真中的行为是否正确；
- 正式构建的 **自动验收**检查部署包及其临时重定位副本是否完整、自洽；
- 最终运行位置的 **首次离线验收**检查候选身份、发布物完整性和 ROS 包解析是否
  适用于实际启动位置与终端环境。

三者缺一不可。`--verify-deployment-only` 不运行仿真，也不能证明机器人行为正确。

本文使用“提交（commit）”表示 Git 提交对象，使用“带说明标签（annotated
tag）”表示在线传递候选版本身份的不可变发布标签。`collect-host`、
`collect-hardware` 和 `analyze` 是命令行子命令，名称保持原样。

## 3. 开发机上的步骤

### 第一步：准备 Sim2Sim 候选版本

进入开发机上的项目：

```bash
RL_SAR_ROOT=/home/lfr/rl_sar
cd "$RL_SAR_ROOT"
git status --short
```

完成代码、配置和模型修改，并按项目要求运行相应单元测试。LW Sim2Sim 和正式实机部署都使用以下四个 ONNX 模型：

```text
policy/LW/robot_lab/leg_loco/policy.onnx
policy/LW/robot_lab/leg_to_wheel/policy.onnx
policy/LW/robot_lab/wheel_loco/policy.onnx
policy/LW/robot_lab/wheel_to_leg/policy.onnx
```

它们分别负责腿式运动、腿转轮、轮式运动和轮转腿。Sim2Sim 还需要开发机上存在 LW 的 MuJoCo 场景，例如：

```text
src/rl_sar_zoo/LW_description/mjcf/scene.xml
```

### 第二步：在开发机编译 Sim2Sim

加载 ROS 2 环境并执行正常的开发构建：

```bash
source /opt/ros/humble/setup.bash
cd "$RL_SAR_ROOT"
./build.sh
```

第一次构建建议编译整个工作区。也可以使用 `./build.sh rl_sar` 做增量构建；
该命令会按 package manifest 自动包含 `rl_sar` 的工作区依赖，而不是只选择
一个可能缺少依赖的包。

LW 实机启动还需要工作区中的 IMU 驱动及其 ROS 串口库。首次构建后应确认三
个包都来自当前开发安装目录：

```bash
source "$RL_SAR_ROOT/install/setup.bash"
ros2 pkg prefix serial
ros2 pkg prefix fdilink_ahrs
ros2 pkg prefix rl_sar
```

仓库只保留 `./build.sh` 作为开发构建入口。不带包名时构建整个工作区；指定
`fdilink_ahrs` 或 `rl_sar` 时，colcon 会先构建其声明的 `serial` 和 IMU
依赖。首次运行会自动检查并安装缺失的 Debian/Ubuntu、ROS、推理和仿真依赖，
可能请求 sudo 权限和网络访问；具体清单见
[README.md](../README.md#获取代码与依赖)。

`build.sh` 的项目参数如下，可用 `./build.sh --help` 查看同一清单：

| 形式 | 含义 |
| --- | --- |
| `./build.sh [PACKAGE_NAMES...]` | 不带包名时构建整个 ROS 2 工作区；带包名时构建指定包及其依赖 |
| `./build.sh -c` / `./build.sh --clean` | 经确认后清理全部工作区构建产物 |
| `./build.sh --clean PACKAGE_NAMES...` | 经确认后清理指定包及其反向依赖的 ROS 2 构建/安装产物 |
| `./build.sh -h` / `./build.sh --help` | 显示用法后退出 |

`--clean` 是显式清理操作。无包名时会删除整个 `build/`、`install/` 和日志目录；
带包名时会先验证包名并列出该包及所有反向依赖，确认后只删除这些包各自的
`build/<package>`、`install/<package>`，保留共享日志、其它包和源码中的标准
`package.xml`。例如 `./build.sh --clean serial` 会同时列出并清理依赖
`serial` 的包，避免留下与新依赖不一致的旧二进制。

ROS 2 开发工作区使用 Colcon isolated install，以提供明确的包级删除边界。

交付前的 Sim2Sim 验证必须使用不带包名的完整工作区构建；它不能代替本文后续由
`build_lw_deployment.sh` 生成和验收的正式部署包。

在 Jetson 上，`./build.sh` 会自动检查 Linux/aarch64、L4T/Tegra 和 Jetson
CUDA 标志，并在日志中输出 `Jetson mode: true`。只有容器等环境隐藏了这些
标志时才使用 `export IS_JETSON=true`；该强制模式在非 Linux/aarch64 主机上
会拒绝运行。普通 x86_64 和可明确排除 Jetson 的 aarch64 环境无需设置变量，
必要时可用 `IS_JETSON=false` 显式禁用。

本项目当前已验证的 Jetson Orin NX 部署基线为 JetPack 6.2.2、Ubuntu 22.04 和
ROS 2 Humble；升级 JetPack、L4T、Ubuntu 或 ROS 前必须重新完成构建与部署验收。
Jetson 所有 C++ 构建只下载并链接 ONNX Runtime，并使用 Linux aarch64
归档，也不启用 CUDA/TensorRT 推理。已有推理库会先校验 ELF 架构，不能把 x86_64
开发机的 `library/inference_runtime` 复制到 Jetson 使用。

推理运行时下载只支持两种已审查组合：Linux x86_64 和 Linux aarch64
的 ONNX Runtime 1.22.0。版本、精确
官方 HTTPS URL 和归档 SHA-256 固定在
`scripts/inference_runtime_archives.json`。下载器先验证完整归档摘要，再在隔离
目录解压并验证结构和 ELF 架构；全部成功后才替换结构损坏的旧目录。一个结构
有效但缺少匹配来源证明、版本更高或摘要不同的现有运行时不会被自动覆盖，构建会
停止并要求把升级作为单独变更审查。该 C++ ONNX 运行时独立于用户
Python 环境中安装的 `torch`/`onnxruntime`；Python `torch` 仍可用于离线
执行器模型训练和评估，不是 C++ 编译依赖。

这个开发构建与后面的正式部署构建用途不同：

- `./build.sh` 生成开发机上的 Sim2Sim 程序；
- `build_lw_deployment.sh` 在部署机生成实机正式运行版本。

维护 C++ 代码还必须通过独立的严格警告门禁：

```bash
scripts/validate_lw_strict_build.sh
```

该脚本使用全新临时目录，以 `-Wall -Wextra -Wpedantic -Werror` 完整构建维护
目标并运行全部 CTest。MuJoCo simulate、joystick 和硬件 SDK 等 vendored 依赖
使用独立编译目标与系统头边界；门禁不会修改或全局豁免维护代码的警告。
默认并行度为 2；需要调整时必须传入正整数环境变量，例如：

```bash
LW_STRICT_BUILD_JOBS=4 scripts/validate_lw_strict_build.sh
```

### 第三步：在开发机完成 Sim2Sim 验证

构建成功后，在开发机启动 LW 的 MuJoCo Sim2Sim：

```bash
source /opt/ros/humble/setup.bash
source "$RL_SAR_ROOT/install/setup.bash"
ros2 run rl_sar rl_sim_LW
```

Sim2Sim 默认不创建高频 Plot publisher、wall timer 或控制快照缓冲。如需临时观察
既有 `/LW_joint_states` payload，可显式启用默认 100 Hz 的绘图遥测：

```bash
ros2 run rl_sar rl_sim_LW --enable-plot
```

也可指定 1–200 Hz 的整数频率：

```bash
ros2 run rl_sar rl_sim_LW --enable-plot --plot-rate-hz 50
```

`--plot-rate-hz` 未与 `--enable-plot` 同时使用，或取值缺失、非整数、超出范围、
重复且冲突时，Sim2Sim 会明确拒绝启动。常规观察建议 50 Hz，默认 100 Hz 用于
普通诊断，200 Hz 仅用于短时逐控制周期分析。该开关只控制 Sim2Sim Plot，不影响
100 ms operator-status 输出，也不改变真机的独立调试 publisher。

这里的 Plot 功能只发布 `/LW_joint_states`，不会在 Sim2Sim 进程内写入 rosbag。
需要保存数据供 PlotJuggler 离线分析时，保持 Sim2Sim 运行，并在另一个终端中
执行：

```bash
mkdir -p bags
bag_dir="bags/lw_sim_$(date +%Y%m%d-%H%M%S)"
ros2 bag record \
    --output "$bag_dir" \
    /LW_joint_states
```

采集结束时先在录包终端按 `Ctrl+C`，等待 rosbag2 完成索引并写入
`metadata.yaml`；不要直接关闭终端或强制杀死 recorder。随后确认 bag 内容并
启动 PlotJuggler：

```bash
ros2 bag info "$bag_dir"
ros2 run plotjuggler plotjuggler
```

在 PlotJuggler 中选择 ROS 2 Bag 数据加载器，打开 `$bag_dir` 对应的 bag 后即可
从 `/LW_joint_states` 中选择所需曲线。记录命令显式限定该话题，不会把工作区内
其它 ROS 2 话题一并写入。

Sim2Sim 默认直接读取编译时仓库根目录下的 `policy/`。该绝对路径
会写入可执行文件，`build/` 和 `install/` 不是默认策略来源。如需改用
其他目录，传入只包含四套 ONNX 主策略的策略根：

```bash
ros2 run rl_sar rl_sim_LW \
    --policy-root /absolute/path/to/policy
```

Sim2Sim 的每个关节始终使用 MuJoCo PD+前馈力矩。`--policy-root`、
`--enable-plot` 和 `--plot-rate-hz` 可在同一条
Sim2Sim 命令中组合。已跟踪的 `.pt` 执行器模型仅供 Python 离线训练/
评估工具使用，`rl_sim_LW` 不加载它们。编译后如果移动仓库，需重新
构建或显式指定新策略根。

关闭 MuJoCo 窗口或按 `Ctrl+C` 都会进入同一条正常线程关闭路径：先请求渲染循环
退出，再关闭 ROS，并按顺序等待业务线程和物理线程结束。`SIGINT` 在任何工作
线程创建前被阻塞，由专用可联结线程同步等待；重复按键只产生一次关闭请求，
不会从异步信号处理器访问 MuJoCo 对象。启动期间收到的请求会被锁存，并在
Sim2Sim 对象构造完成后立即执行。

至少验证以下内容：

- 仿真能够正常启动，模型和 YAML 没有加载错误；
- 初始姿态和关节方向正确，机器人静止时没有明显异常动作；
- `leg_loco` 和 `wheel_loco` 都能正常进入并响应控制命令；
- `leg_to_wheel` 和 `wheel_to_leg` 两个转换能够完整执行；
- 状态切换后关节位置、速度和力矩没有明显跳变或持续发散；
- 连续运行期间没有崩溃、NaN、越界或控制周期异常；
- 本次准备交付的四个 ONNX 和配置与仿真实际加载的文件一致。

如果 Sim2Sim 暴露问题，应回到第一步修改并重新验证。只有 Sim2Sim 通过后，候选版本才能交给部署机进行实机实验。

> [!NOTE]
> Sim2Sim 只能降低实机风险，不能替代实机安全措施。仿真通过后，部署机仍必须完成正式构建的自动验收、最终运行位置的首次离线验收和机械安全检查。

### 第四步：提交 Sim2Sim 验证通过的内容

构建脚本只读取指定 Git 提交中的文件。没有提交的代码、配置和模型不会进入正式部署版本。

提交完成后检查：

```bash
git status --short
git show --stat --oneline HEAD
```

如果 `git status --short` 仍显示文件，应确认它们是否属于本次发布。不要为了让状态变干净而误删与本次发布无关的用户文件。

提交之后不要再修改本次发布使用的代码、模型或配置；如果又有修改，必须重新完成 Sim2Sim 并生成新的提交。

### 第五步：用在线 Git 标签交付已验证提交

两台机器不需要人工复制 40 位完整哈希。开发机给已完成测试和
Sim2Sim 的 `HEAD` 创建一个简短、唯一的 annotated tag，再推送到部署机
可访问的 Git 远程。下面以 `origin` 为远程名；如项目使用其他远程名，
应在开发机和部署机上同步替换。

先为本次候选版选择新标签名。示例中的日期和序号每次发布都应更新：

```bash
RELEASE_TAG=lw-release-20260816-01

# 两条命令都应无输出；有输出说明标签已被使用，应改用新名称
git tag --list "$RELEASE_TAG"
git ls-remote --tags origin "refs/tags/$RELEASE_TAG"
```

确认名称未被使用后，将标签绑定到当前已验证提交，本地解析标签并
核对，然后只推送该标签：

```bash
SOURCE_COMMIT=$(git rev-parse HEAD)
git tag -a "$RELEASE_TAG" "$SOURCE_COMMIT" \
    -m "LW deployment candidate $RELEASE_TAG"

TAG_COMMIT=$(git rev-parse "${RELEASE_TAG}^{commit}")
test "$TAG_COMMIT" = "$SOURCE_COMMIT"
git show --stat --oneline "$TAG_COMMIT"

git push origin "refs/tags/$RELEASE_TAG"
```

annotated tag 记录标签名、说明和目标提交；它用于便于传递及固定候选版
身份，不代替完整哈希和部署清单校验。团队应将已发布标签视为不可变：
不使用 `git tag -f`，不强制推送，也不删除后重建同名标签。如果发布资源又有
修改，必须重新测试、Sim2Sim、提交，并使用新序号创建新标签。

开发机交付时只需把 `RELEASE_TAG` 的值和 Sim2Sim 结果交给部署机。
完整 `SOURCE_COMMIT` 由两台机器分别从同一标签自动解析，不需要人工输入。
开发机到这里完成正式交付准备；后续部署包构建和实机实验在部署机完成。

## 4. 部署机上的准备步骤

### 第一步：进入部署机自己的项目

```bash
RL_SAR_ROOT=/home/lfr/rl_sar
cd "$RL_SAR_ROOT"
git status --short
```

部署机可以保留自己的未跟踪文件或其他开发内容。后面的构建脚本会为指定提交创建临时干净工作树，不会直接从当前工作目录编译。不过，开始前仍应查看状态，避免误操作用户文件。

### 第二步：按发布标签取得开发机指定的提交

填写开发机提供的简短标签名，从同一 Git 远程精确拉取该标签，再由
Git 在部署机上解析完整提交哈希：

```bash
RELEASE_TAG=lw-release-20260816-01
git fetch origin "refs/tags/$RELEASE_TAG:refs/tags/$RELEASE_TAG"

SOURCE_COMMIT=$(git rev-parse "${RELEASE_TAG}^{commit}")
git cat-file -e "${SOURCE_COMMIT}^{commit}"
git tag -n1 "$RELEASE_TAG"
git show --stat --oneline "$SOURCE_COMMIT"
```

精确 refspec 只取得指定标签。如部署机已有同名但指向不同对象的标签，
Git 会拒绝覆盖；此时不得使用 `--force`，应停止并确认发布标签是否被违规
移动。`git cat-file` 没有输出且返回成功，表示部署机已经取得标签对应的
提交。应把 `git tag -n1` 显示的标签说明和 `git show` 显示的提交概要与
开发机的交付记录对照。

构建脚本可以直接构建 `SOURCE_COMMIT`，因此不需要执行 `git checkout`，也
不需要切换部署机当前分支。标签只负责在线传递候选版身份；后续构建、
部署清单和验收仍使用解析出的完整提交哈希。

### 第三步：检查部署机本地依赖

加载部署机安装的 ROS 2：

```bash
source /opt/ros/humble/setup.bash
```

第 5 节的 `build_lw_deployment.sh` 会先确认项目内 ONNX Runtime 目录存在，
再使用候选提交中的 `validate_inference_runtime.sh` 按当前机器架构检查其结构、
共享库和 ELF 架构；只有校验通过才会创建部署输出并开始编译。因此正式流程不需要
在这里手动重复运行该校验脚本。

还需要保证 `git`、`cmake`、`colcon`、C++ 编译器及
[README.md](../README.md#获取代码与依赖) 中的依赖可用。

> [!NOTE]
> 正式部署清单会校验 LW 可执行文件、模型、配置，以及项目自带的
> `fdilink_ahrs`、`serial` 和 ONNX Runtime 运行文件。ONNX Runtime 会以
> 普通文件随部署前缀携带；基础 ROS、Python、操作系统动态库和 USB 内核驱动
> 仍由部署机的系统环境提供。

### 第四步：准备串口设备名和访问权限

LW 使用两条彼此独立的串口路径：

| 用途 | 固定设备名 | 访问程序 | 是否使用 ROS `serial` 包 |
| --- | --- | --- | --- |
| 右侧电机板 | `/dev/ttyLegRight` | `rl_real_LW` 内的 LWSDK | 否 |
| 左侧电机板 | `/dev/ttyLegLeft` | `rl_real_LW` 内的 LWSDK | 否 |
| IMU | `/dev/fdilink_ahrs` | `fdilink_ahrs/ahrs_driver_node` | 是 |

当前项目按现场使用要求将匹配到的 IMU 串口设置为 `0777`。规则安装完成后，
普通登录用户访问 `/dev/fdilink_ahrs` 不需要 root 权限，也不要求属于
`dialout` 组。该设置同时允许本机其他用户读写设备，部署机应限制非授权账户
登录和运行程序。

先用实际枚举出的设备节点确认 IMU 的 USB 属性；下面的 `/dev/ttyUSB0` 只是
示例，必须替换为本机设备：

```bash
udevadm info --query=property --name=/dev/ttyUSB0 \
    | grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT'
```

确认设备确实匹配仓库脚本记录的 CP2102、CH9102 或 CH340 型号后，才可安装
IMU 规则：

```bash
sudo "$RL_SAR_ROOT/src/fdilink_ahrs_ROS2/wheeltec_udev.sh"
```

该脚本只建立 `/dev/fdilink_ahrs` 并将设备权限设置为 `0777`，不会在编译或
离线验收时自动运行。写入 `/etc/udev/rules.d` 和刷新 udev 仍属于系统管理
操作，因此安装规则时必须使用 `sudo`；安装后的普通串口访问不需要 root 或
`dialout`。CH340 通常没有可用于区分同型号设备的唯一序列号；
如果主机连接了多个 CH340，不应直接使用该通用规则，而应先制定能唯一识别
目标 IMU 的现场规则。

仓库目前没有足够的 USB VID、PID 和序列号信息来安全生成
`/dev/ttyLegRight`、`/dev/ttyLegLeft`。部署人员必须先分别读取两块电机板的
实际属性，再用不同的唯一序列号建立稳定别名；不要依赖可能随插拔顺序变化的
`/dev/ttyUSB0`、`/dev/ttyUSB1`，也不要把占位 VID/PID 直接写入系统规则。

重新插拔设备后，只做节点、权限和链接目标检查，不打开串口：

```bash
for device in /dev/ttyLegRight /dev/ttyLegLeft /dev/fdilink_ahrs; do
    test -e "$device" && test -r "$device" && test -w "$device"
    readlink -f "$device"
done
```

## 5. 在部署机生成正式运行版本

构建和验收按以下频率执行：

| 时机 | 必须执行的流程 |
| --- | --- |
| 每个新候选版本 | 完整执行本节构建；构建脚本自动验收原前缀及临时重定位副本 |
| 候选版本放到最终运行位置后首次使用 | 在新终端完整执行第 6 节一次 |
| 同一份未变化且已验收部署的日常重复启动 | 执行第 7 节日常检查；无需重新构建或重复整段第 6 节 |

发布提交或标签、模型、配置、ONNX Runtime 发生变化，或者重新生成了新的部署
输出前缀，都构成新的候选版本，不能沿用旧候选版本的验收结果。部署前缀被复制
或移动到另一个最终位置，以及 ROS、操作系统动态库等运行环境发生变化时，也
必须按第 6 节重新验收。

继续在部署机的 `/home/lfr/rl_sar` 中执行：

```bash
SHORT_COMMIT=$(git rev-parse --short "$SOURCE_COMMIT")
DEPLOY_PREFIX="$RL_SAR_ROOT/build/lw_deployments/$SHORT_COMMIT"

src/rl_sar/scripts/build_lw_deployment.sh \
    "$DEPLOY_PREFIX" \
    "$SOURCE_COMMIT"
```

脚本的位置参数形式是 `<empty-output-prefix> [commit]`。实现中省略提交时
会使用 `HEAD`，但正式部署必须显式传入已经完成 Sim2Sim 验证的完整
`SOURCE_COMMIT`，不允许依赖当前分支或工作树状态。输出前缀必须不存在或为空。

脚本会自动完成以下工作：

1. 从 `SOURCE_COMMIT` 创建临时、干净的源码工作树；
2. 初始化该提交锁定的 Git 子模块；
3. 使用 `Release` 和 `LW_PRODUCTION_DEPLOYMENT=ON` 编译；
4. 安装 `serial`、`fdilink_ahrs`、`rl_real_LW`、LW 配置测量工具、五个
   YAML、四个 ONNX、两个状态转换 CSV，以及匹配本机架构的 ONNX Runtime；
5. 生成 schema v4 `manifest.yaml`，记录源码提交、策略资源、项目内
   IMU/串口运行文件以及 ONNX Runtime 版本、架构和库文件 SHA-256；
6. 拒绝关键部署文件中的符号链接；
7. 检查生产可执行文件只通过 `$ORIGIN/onnxruntime` 解析随包推理库，不保留
   项目源码树中的 ONNX Runtime 路径；
8. 自动验收原部署前缀及其临时重定位副本，两个位置都必须通过
   `--verify-deployment-only`。

输出目录必须不存在或为空。脚本不会覆盖以前的部署版本。如果同一个提交需要重新构建，应使用新的带后缀目录，例如 `${SHORT_COMMIT}_02`，并保留必要的旧版本供回滚。

核心产物位于以下位置；为保持简洁，未逐项列出受清单保护的 `package.xml` 和
ament 包索引：

```text
<DEPLOY_PREFIX>/
├── setup.bash
├── lib/libserial.a
├── lib/fdilink_ahrs/ahrs_driver_node
├── lib/rl_sar/lw_config_profiler
├── lib/rl_sar/profile_lw_runtime_config.py
├── lib/rl_sar/rl_real_LW
├── lib/rl_sar/onnxruntime/
│   ├── libonnxruntime.so.1
│   └── libonnxruntime_providers_shared.so
├── share/fdilink_ahrs/
│   ├── launch/ahrs_driver.launch.py
│   └── wheeltec_udev.sh
├── share/rl_sar/launch/rl_real_LW.launch.py
└── share/rl_sar/deployment/LW/
    ├── manifest.yaml
    └── policy/LW/
        ├── base.yaml
        └── robot_lab/
            ├── leg_loco/
            ├── leg_to_wheel/
            ├── wheel_loco/
            └── wheel_to_leg/
```

## 6. 每个候选在最终运行位置的首次离线验收

构建脚本已经在候选生成阶段自动验收原部署前缀及临时重定位副本。每个新候选
放到最终运行位置后，必须在一个没有加载旧部署版本的新终端中完整执行本节一次，
再进行该候选的首次实机启动；如果构建输出目录本身就是最终运行位置，也仍要
完成这次针对目标终端、包解析和系统动态依赖的首次验收。

同一部署前缀保持在已验收的最终位置，发布文件和 ROS、操作系统动态库等运行
环境均未变化时，日常重复启动不要求重新执行整段本节。只要前缀被复制或移动、
部署文件可能被修改、运行环境或依赖发生变化，或者无法确认已有验收记录仍适用，
就必须在下一次启动前重新完整执行本节。

重新设置变量，因为新终端不会保留上一个终端中的变量：

```bash
RL_SAR_ROOT=/home/lfr/rl_sar
cd "$RL_SAR_ROOT"

# 必须与本次部署的发布标签一致
RELEASE_TAG=lw-release-20260816-01
SOURCE_COMMIT=$(git rev-parse "${RELEASE_TAG}^{commit}")
SHORT_COMMIT=$(git rev-parse --short "$SOURCE_COMMIT")
DEPLOY_PREFIX="$RL_SAR_ROOT/build/lw_deployments/$SHORT_COMMIT"

source /opt/ros/humble/setup.bash
source "$DEPLOY_PREFIX/setup.bash"
"$DEPLOY_PREFIX/lib/rl_sar/rl_real_LW" --verify-deployment-only
```

该命令只检查发布物，不会运行 Sim2Sim，不会创建 ROS 节点，也不会初始化手柄、串口或控制线程。它会检查：

- 当前可执行文件是否属于清单记录的源码提交；
- 可执行文件、五个 YAML、四个 ONNX 和两个 CSV 的哈希是否正确；
- 随包 ONNX Runtime 的版本、架构、精确文件集合和 SHA-256 是否正确；
- `libserial`、AHRS 节点、launch、udev 辅助脚本和三个 ROS 包索引的哈希
  是否正确；
- 资源路径是否仍在部署目录内；
- 关键部署文件本身或其路径是否包含符号链接。

只有命令返回码为 `0` 才表示部署包完整性验收通过。它不能证明策略行为正确，因此必须确认该提交已经在开发机完成 Sim2Sim。任何报错都应停止部署，不能通过直接修改部署目录来绕过检查。

查看清单：

```bash
sed -n '1,220p' \
    "$DEPLOY_PREFIX/share/rl_sar/deployment/LW/manifest.yaml"
```

`manifest.yaml` 中的 `source_commit` 应等于开发机交付的完整提交哈希。标准的
部署机本地构建路径已经自动检查三个生产可执行文件的动态依赖，本节不再手动
重复；跨机器复制部署前缀属于例外路径，必须按第 9 节在目标机重新检查。

还必须确认包解析没有回退到旧开发工作区：

```bash
for package in serial fdilink_ahrs rl_sar; do
    test "$(realpath -m "$(ros2 pkg prefix "$package")")" = "$DEPLOY_PREFIX"
done
```

## 7. 在部署机进行实机实验

以下检查每次实机启动都要执行。新候选的首次启动必须已经完成第 6 节；同一份
未变化且已验收部署的日常重复启动只执行本节检查，无需重新构建或手动重复整段
第 6 节。正常 `rl_real_LW` 启动仍会在打开电机板串口和访问硬件前自动校验清单
及资源哈希，因此这个频率区分不会绕过部署完整性保护。

| 操作 | 执行频率或触发条件 |
| --- | --- |
| 正式构建及构建脚本自动验收 | 每个新候选版本一次 |
| 第 6 节最终运行位置离线验收 | 每个候选到达最终位置后首次一次；位置、文件或运行环境变化后重做 |
| 本节日常启动与现场检查 | 每次实机启动 |
| `collect-host`、`collect-hardware`、`analyze` | 首次部署、软硬件环境变化或需要重新确定参数时 |
| IMU 单独验证和调试话题 | 仅受控验证或故障诊断时 |

### 日常实机启动

启动前至少确认：

- 当前 `DEPLOY_PREFIX` 与验收记录一致，仍位于已经验收的最终运行位置；
- 部署文件、基础 ROS 和系统动态依赖等环境没有发生需要重新验收的变化；
- `manifest.yaml` 中的提交与本次交付记录一致；
- 该提交在开发机上的 Sim2Sim 验证记录已确认；
- 机器人型号、关节映射、限位、初始姿态和四个策略版本正确；
- IMU、串口、执行器及手柄连接已分别验证；
- 机器人已可靠吊装或离地，运动范围内无人和障碍物；
- 硬件急停、电机失能方式及现场监护人员均已就位。

每次启动都在新终端中重新加载基础 ROS 和已验收前缀，并做轻量包解析检查；
这不等同于重复第 6 节的完整离线验收：

```bash
source /opt/ros/humble/setup.bash
source "$DEPLOY_PREFIX/setup.bash"

for package in serial fdilink_ahrs rl_sar; do
    test "$(realpath -m "$(ros2 pkg prefix "$package")")" = "$DEPLOY_PREFIX"
done

PYTHONDONTWRITEBYTECODE=1 ros2 launch rl_sar rl_real_LW.launch.py
```

该 launch 只声明三个项目自定义参数：

| 参数 | 默认值 | 详细边界 |
| --- | --- | --- |
| `enable_keyboard:=<boolean>` | `true` | 只接受 `true`/`false`；见下文“真机终端键盘” |
| `enable_debug_publisher:=<boolean>` | `false` | 只接受 `true`/`false`；见下文“可选调试话题” |
| `debug_publish_rate_hz:=<integer>` | `50` | 只接受 1–200；见下文“可选调试话题” |

三个参数可同时指定。通过 `ros2 launch rl_sar rl_real_LW.launch.py --show-args`
可只读核对当前部署包的声明和默认值，不会启动节点。

必须保留 `PYTHONDONTWRITEBYTECODE=1`。正式部署只允许清单绑定的
`rl_real_LW.launch.py`，禁止 Python 在同一目录生成未经验证的
`__pycache__`；省略该环境变量会使节点在离线完整性检查时安全失败。

正常启动会加载 AHRS 驱动并运行 `rl_real_LW`，随后可能访问真实硬件。不要跳过离线验收，也不要用正常启动命令测试部署包是否完整。

### 启动与安全：正式节点的失能边界

`rl_real_LW` 先完成不访问硬件的部署清单和资源哈希校验。只有该校验通过且本次
不是 `--verify-deployment-only` 后，程序才打开左右电机板串口；此时 ROS、终端
键盘、YAML、FSM 和四个 ONNX 模型都尚未初始化。程序必须先向两侧各完整写入
20 个 `motors_disable=true` 包，并启动独立的 5 ms 失能保活，才会继续上述
运行时初始化。初始化期间不会通过该串口路径发送使能或普通控制包。

全部运行资源准备完成后，程序先停止并等待失能保活线程退出，应用已经验证的
运行时 `serial_write_timeout`，再补写一个双侧完整失能包，最后才启动工作循环。
任一串口仅部分初始化，或者 ROS、终端、配置、模型、FSM、内存分配、循环创建或
循环启动失败，都会阻止控制循环运行；已经打开任一串口后，退出路径会先关闭
命令门并尝试有界的最终 20 包失能序列，再释放串口。

这仍不能消除 STM32 上电后的全部使能窗口。不可避免的最早区间从 STM32 上电
开始，一直持续到操作系统装载程序和动态库、完成部署完整性校验、程序打开串口
并把首批失能包完整写入主机内核串口队列。主机完整写入不等于 STM32 已接收或
执行失能，也不是硬件回执。因此正式启动前仍必须可靠吊装或离地，保持运动范围
隔离，并由现场人员掌握物理急停；不能用本启动顺序替代这些措施。

### 操作员输入：真机终端键盘

真机 launch 默认设置 `enable_keyboard:=true`。`rl_real_LW` 会直接打开启动该
launch 的控制终端 `/dev/tty`，切换为非规范、无回显的非阻塞输入，但保留
`Ctrl-C` 信号。终端配置会在正常退出、启动异常和对象销毁时恢复。键盘读取由
200 Hz 控制线程执行，不会启动另一个线程并发修改 FSM 输入。

数字键 `9` 是现有 FSM 的 `GetDown` 请求。手柄断联后，程序仍会永久锁住该
进程的 Gamepad 输入并把 `x/y/yaw` 置零，但不会清除终端键盘；确认下降路径
安全后仍可在这个终端按 `9` 请求受控趴下。其他数字键和 `P` 等键位仍按当前
FSM 状态解释，不能在机器人未吊装、人员位于运动范围内时试键。

启用键盘却没有可访问的控制终端时，真机节点会在控制循环启动前明确失败，
避免错误宣称存在键盘恢复通道。systemd、容器或其他明确无交互终端的部署必须
显式关闭：

```bash
PYTHONDONTWRITEBYTECODE=1 ros2 launch rl_sar rl_real_LW.launch.py enable_keyboard:=false
```

关闭后不存在终端 `GetDown` 通道，必须在启动前准备独立的受控恢复方式、可靠
机械支撑和物理急停；不得仅因为节点仍在运行就假设手柄断联后可以安全恢复。

### 操作员输入：当前键盘和手柄映射

以下表格描述当前 LW 代码实际使用的映射，不是通用推荐键位。FSM 按键只在表中
列出的状态前提满足时生效；没有满足前提时，按键不会强制跳过起身、趴下或形态
切换过程。首次试键必须保持机器人可靠吊装或离地，并由现场人员掌握物理急停。

#### FSM 和模式切换

| 功能 | 键盘 | 手柄 | 生效状态和说明 |
| --- | --- | --- | --- |
| 腿式起身 | `0` | `A` | 在 Passive、轮式起身或 GetDown 状态请求 `RLFSMStateGetUp_Leg` |
| 进入腿式运动 | `1` | `RB` + 十字键上 | 仅在腿式起身完成后进入腿式 locomotion；腿式 locomotion 中重复请求保持当前状态 |
| 轮式起身 | `2` | `Y` | 在 Passive、腿式起身或 GetDown 状态请求 `RLFSMStateGetUp_Wheel` |
| 进入轮式运动 | `3` | `RB` + 十字键下 | 仅在轮式起身完成后进入轮式 locomotion；轮式 locomotion 中重复请求保持当前状态 |
| 腿式切换为轮式 | `4` | `RB` + 十字键左 | 仅在腿式 locomotion 中启动 `leg_to_wheel` 策略；完成后自动进入轮式 locomotion |
| 轮式切换为腿式 | `5` | `RB` + 十字键右 | 仅在轮式 locomotion 中启动 `wheel_to_leg` 策略；完成后自动进入腿式 locomotion |
| 受控趴下 | `9` | `B` | 起身完成后、任一 locomotion 或形态切换状态中请求 GetDown；Passive 中无动作 |
| 立即转入 Passive 阻尼 | `P` | `LB` + `X` | 任一非 Passive 状态转入 `RLFSMStatePassive`；这是 `Kp=0`、`Kd=5` 的阻尼命令，不等于电机失能或零执行器输出 |
| 切换导航模式标志 | `N` | `X` | 切换 `navigation_mode`；当前 LW 本地速度仅由摇杆输入 |

组合键写法表示先按住肩键，再按面键或推动十字键。`GetDown` 是受控运动，不是
急停；受保护状态下姿态角度越限会进入 S2 阻尼，反馈或控制线程已经不可信时
仍会按 S4 硬失能并退出。操作员应保留物理急停和可靠支撑。

#### 速度指令

| 功能 | 键盘 | 手柄 |
| --- | --- | --- |
| 前向速度 `x` | 无 | 左摇杆纵向：向前为正、向后为负 |
| 横向速度 `y` | 无 | 左摇杆横向：向左为正、向右为负 |
| 偏航速度 `yaw` | 无 | 右摇杆横向：向左为正、向右为负 |
| 三轴速度清零 | 无 | 将已使用的摇杆回中 |

LW 键盘不提供速度控制：`W/S/A/D/Q/E` 和 `Space` 都不会修改 `x/y/yaw`；
`Space` 不会停车，也不是急停。正常人工速度只来自手柄，停止时应将已使用的
摇杆回中；异常情况下使用运行时安全门控、物理急停和可靠机械支撑。手柄速度按
当前策略 `vel_command` 缩放；当前已提交的 LW 策略将横向上限
`vel_command[1]` 设为 `0.0`，因此默认配置下左摇杆横向不会产生 `y` 指令。
进入 S1 输入降级后，运行时会锁住 `x/y/yaw` 为零，但仍保留上表的 FSM 恢复
按键。推理周期会从同一个有效命令快照生成速度和步态观测：只要手柄故障、外部
输入故障或安全监督器使有效速度归零，当帧的 `commands` 和 `gait_phase` 就会
分别固定为 `{0, 0, 0}` 和 `{0, 0}`，不会出现零速度配合运动相位。短暂输入抑制
不会重置内部相位时钟，解除抑制后仍沿原有时间线继续。

#### 仅 Sim2Sim 使用

| 功能 | 键盘 | 手柄 | 说明 |
| --- | --- | --- | --- |
| 重置到腿式初始姿态 | `R` | `RB` + `Y` | 调用 MuJoCo 的 `home_leg` keyframe；真机无此功能 |
| 重置到轮式初始姿态 | `T` | `RB` + `A` | 调用 MuJoCo 的 `home_wheel` keyframe；真机无此功能 |
| 暂停/继续仿真 | `Enter` | `RB` + `X` | 切换 MuJoCo 运行状态；真机无此功能 |

#### 手柄布局和未绑定输入

真机与默认 Sim2Sim 当前都按 `JOYSTICK_1` 布局解释 `/dev/input/js0`：按钮编号
`0/1/3/4` 分别作为 `A/B/X/Y`，`6/7` 作为 `LB/RB`，轴 `0/1/2` 分别作为
`LX/LY/RX`，轴 `6/7` 作为十字键横向/纵向。这里的 `A/B/X/Y` 是程序赋予
Linux joystick 编号的逻辑名称；不同品牌、连接方式或驱动可能报告不同编号，
必须在机器人断电或可靠吊装时先核对实际设备事件，不能只看手柄外壳标识。

除上表外，键盘解析器虽然能识别其他字母、数字、方向键和 `Escape`，手柄层也
能识别单独的 `LB/RB`、十字键、摇杆按下及其他组合，但当前 LW FSM 和适配器
没有为它们绑定动作。尤其是 `M/K`、`LB+A`、`LB+B` 当前不负责电机使能或
失能，单独十字键也不调整步频；不得依据通用 SDK 注释或其他机器人配置推测
这些按键的功能。

### 可选诊断：受控验证 IMU 话题

这一步不是自动化验收的一部分，因为它会真实打开 `/dev/fdilink_ahrs`。只能在
确认 IMU 型号、波特率 `921600`、设备别名和权限正确，并且尚未启动完整
`rl_real_LW` 时单独执行：

```bash
ros2 launch fdilink_ahrs ahrs_driver.launch.py
```

当前 `ahrs_driver.launch.py` 没有声明项目自定义 launch 参数；
`ros2 launch fdilink_ahrs ahrs_driver.launch.py --show-args` 会报告“No arguments”。
不应在这条命令后追加自行猜测的串口或话题参数。

在另一个同样加载当前部署前缀的终端检查：

```bash
ros2 topic echo --once /imu
ros2 topic hz /imu
ros2 topic echo --once /euler_angles
ros2 topic hz /euler_angles
```

确认 `/imu` 时间戳持续更新、姿态/角速度/加速度为有限值，并确认
`/euler_angles` 独立持续发布且两者频率稳定后，用
`Ctrl-C` 停止独立 AHRS 驱动。不要让独立 AHRS 驱动与完整 LW launch 同时
争用同一串口。

完整 LW launch 会在不修改 FDLink 包的前提下，将这两个第三方话题分别重映射为
`/fdilink/raw_imu` 和 `/fdilink/raw_euler`。实机进程内的 guard 只把一次有限的
AHRS 事件作为一次性授权，并只接受授权时效内紧随其后、角速度有限且四元数模长
合理的下一帧原始 IMU；通过后四元数会归一化。启动以来从未出现有效配对时，控制
保持硬失能等待；已经进入 Ready 后可信 IMU 过期，则锁存硬失能并关闭程序。

### 可选诊断：调试话题

正式启动默认不创建高频调试 publisher 或定时器，也不会在控制线程中复制调试
快照。如需临时观察当前/目标关节、IMU 和速度指令，可显式启用默认 50 Hz 的
`/LW_joint_states` 话题；`debug_publish_rate_hz` 只接受 1–200 的整数：

```bash
PYTHONDONTWRITEBYTECODE=1 ros2 launch rl_sar rl_real_LW.launch.py \
  enable_debug_publisher:=true debug_publish_rate_hz:=50
```

即使 publisher 关闭，频率参数仍会在 worker 启动前校验，但不会创建调试资源或
复制快照。该话题仅用于受控调试和绘图；控制线程使用非阻塞交接，争用时丢弃
调试样本。发布端只发布尚未发出的最新控制源帧，每条消息使用同一个控制周期的
完整快照并在发布时刷新时间戳，不会把未变化的旧帧重新标记为新遥测。调试结束
后应恢复默认关闭状态，避免不必要的 ROS 发布负载。

需要保存 `/LW_joint_states` 供 PlotJuggler 离线分析时，在同一控制机的第二个
终端中录制。将 `LW_BAG_ROOT` 替换为容量充足的可写绝对路径；该路径必须位于
`DEPLOY_PREFIX` 之外，禁止向受清单保护的正式部署目录写入 bag：

```bash
LW_BAG_ROOT="/absolute/writable/path/outside/DEPLOY_PREFIX"
mkdir -p "$LW_BAG_ROOT"
bag_dir="$LW_BAG_ROOT/lw_real_$(date +%Y%m%d-%H%M%S)"
ros2 bag record \
    --output "$bag_dir" \
    /LW_joint_states
```

正常采集结束时先在录包终端按 `Ctrl+C`，等待 rosbag2 完成索引并写入
`metadata.yaml`，然后检查 bag 并启动 PlotJuggler：

```bash
ros2 bag info "$bag_dir"
ros2 run plotjuggler plotjuggler
```

在 PlotJuggler 中选择 ROS 2 Bag 数据加载器，打开 `$bag_dir` 对应的 bag 后从
`/LW_joint_states` 中选择所需曲线。记录命令显式限定该话题，不会把其它 ROS 2
话题一并写入。

录制会在控制机上产生磁盘 I/O，开始前应确认可用空间并限制采集时长。rosbag2
是外部订阅者，recorder 异常退出、写入失败或磁盘写满都不会停止机器人，操作者
必须单独监控录制状态。调试 publisher 允许因争用或新帧覆盖而丢弃样本，因此
bag 不代表无损控制周期日志。正常结束应使用 `Ctrl+C` 完成 bag 收尾；机器人
出现异常时必须优先执行物理急停和失能，不能为保存 bag 延迟安全操作。采集结束
后恢复默认关闭 `enable_debug_publisher`。

### 控制循环与安全：默认保护行为

LW 实机控制循环按绝对时间点以 5 ms 周期运行。某次执行过慢时，程序会跳过已经过期的周期，不会为了“补次数”而连续突发执行控制回调。终端状态和周期统计由 ROS 定时器输出，不在 200 Hz 电机命令线程中打印。

正式版本使用其清单内的 `policy/LW/base.yaml`。当前默认值的含义如下：

```yaml
control_loop_cpu: -1
control_loop_realtime_priority: 0
control_loop_require_realtime: false
control_loop_degraded_consecutive_misses: 3
control_loop_degraded_lateness: 0.02
control_loop_fatal_consecutive_misses: 0
control_loop_fatal_lateness: 0.0
```

- `cpu: -1` 表示暂不固定 CPU；`realtime_priority: 0` 表示使用普通的 `SCHED_OTHER` 调度。这是未经部署机测量前的可移植默认值。
- 连续错过 3 个控制周期，或者单次唤醒或执行晚到达到 20 ms，会进入“时序降级”状态。
- 时序降级会永久锁住本次进程的 `x/y/yaw` 为零，但保留 FSM 按钮输入。操作员仍可请求 `GetDown`；处理完现场安全后必须重启进程才能清除该锁存。
- 默认的两个 `fatal` 值为零，表示控制循环晚到只进入上述降级状态，不会自动硬失能。姿态角度越限进入 S2 阻尼；传感器过期、非法最终命令和控制回调异常等不可信故障仍保持硬失能。
- 每个策略输入都携带当前策略代际、单调序号和 `GetState()` 完成时的采集时间。推理线程对每个输入最多运行一次；没有新控制输入时不会用同一旧状态反复生成看似更新的输出。策略输出的年龄从该状态采集时间计算，而不是从推理完成时间计算。当前上限沿用 `3 * dt * decimation`，随已提交的 5 ms/4 配置为 60 ms；达到边界仍有效，超过边界、未来时间或来源回退会锁存 S2。一个仍在 60 ms 窗口内的完整 50 Hz 输出可以由 200 Hz 控制循环保持使用，不要求每个控制周期都有新推理输出。
- 每秒一条 `[Timing] loop_control` 日志会报告平均/最大唤醒晚到、最大截止时间晚到、最大回调执行时间、错过截止时间和跳过周期数。应把实机吊装测试时的这些日志保存到验收记录中。

不要直接修改已经生成的部署目录来调整参数。参数调整应在项目的 `policy/LW/base.yaml` 中完成，经过开发机测试和 Sim2Sim、提交，再由部署机从新提交生成新的部署版本。

### 首次部署或环境变化：吊装配置测量

可靠吊装并保持 FSM 在 Passive 时，可以完成两阶段的部署前测量：第一阶段在
目标 Jetson 上不连接任何硬件，运行与正式部署相同的四个 ONNX、观测/输出路径、
200 Hz 控制调度和 50 Hz 推理；第二阶段只观察真实 IMU 与两块电机板反馈，并
持续发送电机失能包。策略仍会在后台做 shadow inference，但其输出始终被丢弃，
不会进入腿式/轮式 Locomotion 或形态切换状态。

这两阶段可给出以下候选：

- `control_loop_cpu`、显式试验过的 `control_loop_realtime_priority`；
- 控制循环降级的连续丢周期数和单次晚到阈值；
- 左右电机板反馈的 `sensor_timeout`；
- guard 输出中断的 `trusted_imu_timeout`；
- AHRS 一次性授权到下一帧 IMU 的 `imu_ahrs_pair_max_age`；
- 两个串口组成的完整命令包写入截止时间 `serial_write_timeout`。

吊装静态测量不能验证运动负载、接地冲击、电源压降、电机板看门狗和硬失能后的
机械结果。因此工具永远保留
`control_loop_fatal_consecutive_misses: 0`、
`control_loop_fatal_lateness: 0.0`，并保持
`control_loop_require_realtime: false`；这三项只能在单独的物理安全验证和部署权限
评审后人工决定。输出的其他值也只是吊装环境候选，不是自动生效的最终配置。

`profile_lw_runtime_config.py` 及其 `collect-host`、`collect-hardware`、
`analyze` 子命令都支持 `-h`/`--help`。帮助模式在参数解析阶段退出，
不会运行 profiler、初始化 ROS 或访问 IMU、串口和电机。

#### 第一阶段：无硬件 CPU 和调度测量

在目标部署机的新终端中加载本次部署前缀。以下命令不会初始化 ROS、IMU、手柄、
串口或执行器设备：

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

`--cpus allowed` 会逐一测量当前进程亲和掩码允许的逻辑 CPU；也可用
`--cpus=-1,0,1` 显式加入不绑定的对照组。正数实时优先级只应由部署负责人明确
选定后加入，例如 `--realtime-priorities 0,50`。只要报告请求了 CPU 绑定或
正数实时优先级但实际未成功应用，`analyze` 就会拒绝整次分析，不会静默跳过
该报告。每个输出目录必须不存在或为空，工具
拒绝覆盖历史测量。同一轮用于比较的全部 host 报告必须使用相同的
`--duration-seconds`；累计丢周期数只在该条件下参与排序，不能混用不同时长的
历史报告。

`collect-host` 的 `--duration-seconds` 必须为正数，默认 30；
`--realtime-priorities` 接受逗号分隔的非负整数。获得批准后可追加
`--require-realtime`：包装脚本只对正数优先级向 profiler 传递该开关，使
`SCHED_FIFO` 应用失败的该组测量直接终止；优先级 0 仍使用
`SCHED_OTHER`。

#### 第二阶段：吊装硬件观察

硬件阶段必须同时满足：机器人可靠吊装、运动范围隔离、现场人员掌握物理急停、
正常 `rl_real_LW` 已停止、两个电机板串口没有其他读写者。只单独启动 AHRS，
不要启动完整 LW launch：

```bash
ros2 launch fdilink_ahrs ahrs_driver.launch.py
```

在另一个已加载相同部署前缀的终端执行。确认字符串必须逐字匹配：

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

`collect-hardware` 的项目参数和默认值如下：

| 参数 | 默认值 | 约束 |
| --- | --- | --- |
| `--duration-seconds` | `60` | 必须为正数 |
| `--cpu` | 无，必填 | `-1` 表示不绑定，其它值必须为非负整数 |
| `--realtime-priority` | `0` | 必须为非负整数 |
| `--require-realtime` | 关闭 | 只能与正数 `--realtime-priority` 同时使用 |
| `--right-port` | `/dev/ttyLegRight` | 右电机板串口 |
| `--left-port` | `/dev/ttyLegLeft` | 左电机板串口 |
| `--imu-topic` | `/imu` | 独立 AHRS 驱动的原始 IMU 话题 |
| `--ahrs-topic` | `/euler_angles` | 独立 AHRS 驱动的欧拉角话题 |
| `--confirmation` | 无，必填 | 必须与上述确认字符串逐字匹配 |

只有稳定设备别名或话题配置已被单独评审，且报告中的测量对象与最终
部署完全一致时，才可覆盖端口或话题。不得为绕过设备准备错误而临时指向
其它串口或话题。添加 `--require-realtime` 时必须同时使用已批准的正数
`--realtime-priority`，否则 profiler 会明确拒绝启动。

**明确结论：执行上述 `collect-hardware` 命令的整个测算过程中，测算程序不会使能
电机。** 它不会向左右电机板串口发送电机使能包或普通控制包；四个策略只用于
shadow 推理和耗时测量，其生成的 Passive/策略命令会被丢弃，不会进入串口。测算
程序在初始化、测算、退出三个阶段唯一允许写入电机板串口的命令均为
`motors_disable=true`，其中测算期间由独立的 5 ms 线程持续发送失能保活。

该模式不会创建手柄或键盘输入，也不会允许 Passive/shadow policy 命令进入串口。
它会先打开左右电机板串口并确认 20 个 `motors_disable=true` 包均完整写入，随后
启动独立的 5 ms 失能保活线程；只有失能写入和保活启动成功后才会预加载四个模型、
启动 ROS 观察和参数测算。初始化、测算和退出阶段不存在非失能命令发送路径；
任一失能包写入不完整都会终止测算，退出前仍尝试最终 20 个失能包。报告记录
初始失能写入与保活证明、原始 IMU、有效 AHRS、guard 接受的可信 IMU、配对时延、
左右电机板首帧等待、相邻有效帧间隔、结束时帧龄、串口写耗时与失败次数。任何
失能证明缺失、来源未出现、配对或采样不足、串口写失败或四个策略未完整运行，
都会使候选分析失败。命令结束后停止独立 AHRS 驱动。

当前电机板反馈协议没有单独的“失能已执行”回执位，因此上述证明只表示上层失能
包已完整写入两个串口且程序没有发送非失能命令，不能替代 STM32 侧状态确认或
物理急停。换言之，可以确认“测算程序不会发出使能命令”，但不能仅凭该报告确认
“STM32 已经执行失能且电机物理上处于失能状态”。底层是否实际进入失能状态仍应
由固件行为和现场安全措施保证，整个测算期间必须保持可靠吊装、运动范围隔离和
物理急停就位。

#### 生成仅供评审的候选

下面四个 `max-safe` 值不是脚本测出来的性能值，而是风险评估预先确定的硬上限；
必须由负责机械与控制安全的人员给出。先把下面四个变量的占位内容替换为评审
确定的有限正数，再执行分析命令：

```bash
MAX_SAFE_SENSOR_TIMEOUT_MS=REPLACE_WITH_REVIEWED_POSITIVE_MS
MAX_SAFE_TRUSTED_IMU_TIMEOUT_MS=REPLACE_WITH_REVIEWED_POSITIVE_MS
MAX_SAFE_IMU_AHRS_PAIR_AGE_MS=REPLACE_WITH_REVIEWED_POSITIVE_MS
MAX_SAFE_CONTROL_GAP_MS=REPLACE_WITH_REVIEWED_POSITIVE_MS

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

`--reports` 接受一个或多个报告路径；本流程要求同一部署的全部 host 报告和
一份 hardware 报告。四个 `--max-safe-*-ms` 参数在通用接口中可省略，省略时
相应候选会保持现有值并标记待评审；但本正式实机候选流程必须全部填写
为由机械与控制安全负责人确定的有限正数。`0`、负数、`NaN` 和无穷值均
会被拒绝。`--minimum-hardware-samples` 必须为正整数，默认 1000。
`--output` 指定的文件不得已经存在。

分析器按截止时间丢失、最大晚到、最大执行时间和推理尾延迟排序 host 报告；电机
反馈和可信 IMU 候选覆盖相应的 P50/P99.9/最大帧间隔及结束帧龄，配对候选覆盖
P99.9/最大 AHRS 到后续 IMU 的时延；串口候选覆盖 P99.9/最大写耗时，并且必须
小于一个 5 ms 控制周期。`policy/LW/base.yaml` 中新增的两个 IMU 参数暂时保留
原 100 ms 行为作为未测量占位值，不代表已批准的实机安全窗口。若不提供某个
安全上限，对应现有值会保持不变并标记为需要人工测量或评审。

每份 profiler 报告使用 schema v3，记录完整源码提交、主机节点/系统/内核/架构、
规范化策略根、每策略测量时长，以及固定顺序的 11 个获批策略资产及 SHA-256。
四个策略记录也必须各出现一次并保持批准顺序。分析器只接受同一提交、同一主机、
同一策略根和同一资产摘要的报告；host 报告时长必须完全相同，hardware 报告可
使用独立时长但不会参与 host 排名。schema v1、重复或乱序策略、混合部署、混合
host 时长以及模式与硬件证明矛盾的报告都会被拒绝，必须从当前部署重新采集。
schema v2 报告不含独立 AHRS、可信 IMU 和配对时延证据，不能用于本版本候选分析。

`--base-yaml` 必须正好是报告中策略根下的 `LW/base.yaml`，且摘要与报告一致。
分析器在读取和写出前后都会复核 base、11 个策略资产和输入报告未发生变化；不能
用另一提交的源码树或手工复制的旧 `base.yaml` 生成候选。

`candidate-review.json` 是 JSON（也是 YAML 1.2 可读的映射），包含
`review_only: true`，不会修改输入 `base.yaml` 或部署目录。候选同时复制部署
身份、base 路径与摘要、每个输入报告的路径/摘要/模式/时长，以及排序分数和选中
来源，供评审人员逐项追溯。必须人工查看报告、在源码树的
`policy/LW/base.yaml` 中单独修改、重新执行 Sim2Sim 和测试、提交，再从新提交
生成新部署版本。不得把候选文件直接覆盖到当前部署包。

### 控制循环与安全：可选的 CPU 固定和实时优先级

只有在目标部署机上完成负载测量后，才应设置 `control_loop_cpu` 或正数的 `control_loop_realtime_priority`。先用 `lscpu` 确认可用 CPU，再确认运行账户具备设置 `SCHED_FIFO` 的权限；具体授权方式应遵守部署机的 systemd 或安全配置，不要仅为绕过权限错误而以 root 身份运行整个控制程序。

当设置了正数优先级但 `control_loop_require_realtime: false` 时，权限不足会记录警告并回退到 `SCHED_OTHER`；设为 `true` 后，实时调度失败会使控制回调在首次执行前拒绝启动。只要显式配置了 CPU 编号，CPU 固定失败就会拒绝启动，不受该开关影响。这些配置必须先在吊装状态下验收。

`control_loop_fatal_consecutive_misses` 和 `control_loop_fatal_lateness` 只应在取得部署机时序数据、确认电机板端看门狗行为并决定硬失能策略后显式启用；后一个值的单位是秒。启用任一严重阈值后，达到阈值会进入下文的 S4：锁死命令门、发送约 100 ms 的电机失能包并请求 ROS 关闭，机器人不会自动执行受控趴下。

### 控制循环与安全：实机运行时的安全分级

程序不再把所有异常都当成“立即断力并退出”。处理强度由故障发生在哪个环节、此时控制数据是否还可信来决定：

| 级别 | 触发情况 | 程序做什么 | 现场处理 |
| --- | --- | --- | --- |
| S0 诊断 | 串口解析错误，但新鲜有效反馈仍持续到达；力矩保护告警 | 记录告警，不改变控制 | 观察错误计数；若有效数据随后过期，会按 S4 处理 |
| S1 输入降级 | 手柄断开/手柄线程异常，或控制循环连续错过 3 周期/单次晚到 20 ms | 锁住速度指令为零，不断电、不关闭 ROS；时序降级时保留 FSM 按钮 | 可在判断安全后请求 `GetDown`；重启才能恢复该输入源 |
| S2 受控阻尼 | 推理线程异常，策略输入不完整、来源回退、未来时间或过期，策略 action/输出非法，策略输出不完整或过期，受保护状态下横滚或俯仰绝对值超过 75° | 不再接受策略输出；在下一个可执行的控制周期把命令覆盖为 `Kp=0`、`Kd=5`、速度与前馈力矩为零；若是控制线程当周期发现输出过期或姿态角度越限，则在当周期直接覆盖旧命令。由控制线程转入 `RLFSMStatePassive` | 扶稳机器人并停止实验；该故障不自动恢复，排查后重启 |
| S3 硬失能 | 电机板报告硬件故障 | 关闭命令门并发送 20 次失能包；不主动关闭 ROS，便于保留诊断 | 立即支撑机器人、使用物理急停/断电，不得原地恢复；排除硬件故障后重启 |
| S4 硬失能并退出 | 控制线程异常、传感器过期/反馈读取失败、反馈非法、FSM 丢失、最终命令非法/发送不完整、显式启用的致命时序阈值 | 关闭命令门，发送 20 次失能包，然后请求 ROS 关闭 | 立即支撑机器人并使用物理急停/断电；不要期待自动趴下 |

姿态角度保护仅用于腿式行走、轮式行走、腿转轮和轮转腿。`GetDown`、两种
`GetUp` 和 `Passive` 均不检查 75° 角度限制，但仍检查反馈有效性。姿态越限
会在发现越限的同一控制周期转入 `Passive` 并发送阻尼命令，程序保持运行。该 S2
故障保持锁存，恢复角度或按起身键均不能恢复运动，须排查后重启进程。

启动阶段如果串口初始化失败、初始失能序列不完整、初始化期间的 5 ms 失能保活
失败、运行时交接失能不完整或循环启动失败，程序会关闭命令门、停止已经启动的
循环，并尝试最终 20 包失能序列后中止启动。正常退出仍是先关闭命令门、停止并
等待全部循环退出，再发送最终 20 包失能序列。

### 控制循环与安全：主机保护边界

上述等级是主机程序的决策，不等于已经证明电机物理上失能。当前主机只能证明它已经调用串口发送失能帧；尚未在本项目中证明以下硬件事实：

- 电机板对 `motors_disable` 的确切执行时延、回执与故障时行为；
- 串口断线或主机死机后，板端看门狗是否会在有界时间内失能；
- `Kd=5` 的 Passive 阻尼在所有实机姿态和负载下是否都不会导致二次危险；
- 75° 姿态阈值与受控阻尼的组合是否适合实机的每一种受保护状态。

因此，首次实机验收前必须在机器人悬空、轮子和关节无法触及人员的条件下，逐项注入 S1–S4 故障，测量实际电机响应、串口失效和板端看门狗。在得到这些结果前，物理急停、可靠支撑和安全距离仍是必需条件，不能用软件分级代替。

## 8. 升级和回滚

### 升级

每次升级都重复同一流程：

1. 开发机完成修改、单元测试和 Sim2Sim；
2. 开发机提交验证通过的内容，创建并推送新的不可变发布标签，将
   Sim2Sim 结果和标签名交给部署机；
3. 部署机精确拉取该标签并自动解析完整提交哈希；
4. 部署机构建到新的 `build/lw_deployments/<提交短哈希>/`；
5. 部署机完成最终运行位置的首次离线验收和实机安全检查；
6. 加载新目录的 `setup.bash` 后进行实机实验。

不要覆盖旧部署目录，也不要在部署目录中替换模型、配置或可执行文件。需要修改任何受清单管理的文件时，应提交修改并生成新的部署版本。

### 回滚

回滚不需要切换整个项目的 Git 分支。打开新终端，将 `DEPLOY_PREFIX` 改为上一个已经验收的目录，重新执行：

```bash
source /opt/ros/humble/setup.bash
source "$DEPLOY_PREFIX/setup.bash"
"$DEPLOY_PREFIX/lib/rl_sar/rl_real_LW" --verify-deployment-only
```

验收和安全检查通过后，再执行实机启动命令。

## 9. 例外路径：从开发机复制部署版本

标准路径是在部署机本地构建。只有开发机和部署机的 CPU 架构、操作系统、ROS
版本及 C/C++ ABI 已确认兼容时，才考虑在开发机构建后复制完整部署前缀。
ONNX Runtime 已包含在部署前缀中，不再要求目标机另有项目内推理库路径，但其
架构仍必须匹配目标机。

不能只复制 `rl_real_LW` 或 `deployment/LW`，必须复制整个 `<DEPLOY_PREFIX>`。
复制到部署机后，仍必须重新执行 `--verify-deployment-only` 和 `ldd` 检查；这是
跨机器例外路径的目标机兼容性检查，不是标准本机构建路径的日常重复步骤。

生产可执行文件使用相对于自身的 ONNX Runtime 搜索路径；构建脚本会把部署前缀
复制到临时新位置并再次验收。基础 ROS、Python 和系统 ABI 仍可能不同，因此
只要不能确认两台机器的环境兼容，就应使用本文推荐方案：在部署机自己的
`rl_sar` 项目中按指定提交本地构建。

不要把完整开发仓库中的 `library/inference_runtime` 从 x86_64 主机复制到
Jetson。即使目录结构完整，其 ELF 架构仍不兼容；当前构建会在 CMake 前明确
拒绝这种运行时。

## 10. 常见问题

### 为什么部署机已有完整项目还要生成部署版本

完整项目会继续变化，普通构建目录也可能残留旧文件。部署版本把一次正式运行使用的源码提交、二进制、四个模型和配置绑定在一起，并允许在启动前验证。它是完整项目中的“已验收运行版本”，不是项目代码的替代品。

### 部署机的离线验收能代替开发机 Sim2Sim 吗

不能。`--verify-deployment-only` 只确认文件没有缺失、篡改或版本混用，不执行策略推理闭环和机器人运动。策略行为、四种状态和两种转换必须先在开发机用 `rl_sim_LW` 完成 Sim2Sim 验证。

### 部署机必须切换到开发机的分支吗

不需要。只要部署机的 Git 仓库已经取得指定提交，构建脚本就能直接从该提交创建临时工作树。部署机当前分支和未提交文件不会进入部署版本。

### 输出目录不是空目录

脚本会报 `Output prefix must not exist or must be empty`。请选择新的版本化目录，不要覆盖已验收版本。

### ONNX Runtime 缺失

构建前脚本报 `ONNX Runtime dependency is missing` 时，检查部署机当前项目中的
`library/inference_runtime/onnxruntime` 是否完整。部署生成后若随包库缺失，
manifest 校验或动态加载器会在 ROS、串口和电机初始化前拒绝运行；不要从其他
部署目录手工补文件，应从预期提交重新生成整个部署版本。

如果脚本报告现有运行时缺少批准的归档来源或与固定清单不符，不要删除来源检查、
伪造 `origin.json` 或让脚本自动降级覆盖。确认该目录是否来自其他版本；需要升级
时应单独更新版本、URL、归档摘要，完成完整构建和 Sim2Sim 后再生成部署包。

### ONNX Runtime 架构不匹配

校验脚本会报告期望和实际 ELF `Machine`。删除或移出从其他 CPU 架构复制来的
`library/inference_runtime/onnxruntime`，然后在 Jetson 项目中重新运行
`./build.sh`，由下载脚本取得 Linux aarch64 版本；不要通过跳过校验继续链接。

### 临时工作树不干净或子模块初始化失败

确认指定提交中的 `.gitmodules` 和子模块提交可访问。构建依赖的源码或模型必须在指定提交中，不能依赖当前工作区的未跟踪文件。

### 清单、哈希或源码提交不匹配

部署目录可能被修改、复制不完整，或混入了其他版本的二进制和资源。停止使用该目录，从预期提交重新生成部署版本。

### ROS 找到了另一个工作区

关闭已经加载其他工作区的终端。在新终端中先加载基础 ROS，再加载目标部署目录的 `setup.bash`，然后确认：

```bash
ros2 pkg prefix serial
ros2 pkg prefix fdilink_ahrs
ros2 pkg prefix rl_sar
```

输出应与当前 `DEPLOY_PREFIX` 一致。

### 动态库缺失或跨机器 ABI 不兼容

标准的部署机本地构建会自动拒绝 `not found`。如果按第 9 节跨机器复制部署
前缀，应在目标机使用 `ldd` 查找缺失项并确认 ABI 兼容；ONNX Runtime 必须解析
到当前部署前缀。其他缺失项应通过部署机安装 ABI 兼容的 ROS 和系统依赖解决，
不得用 `LD_LIBRARY_PATH` 指向项目源码树来绕过随包 ONNX Runtime 校验。

## 11. 相关文件

- 构建脚本：`src/rl_sar/scripts/build_lw_deployment.sh`
- 配置采集器：`src/rl_sar/src/lw_config_profiler.cpp`
- 配置候选分析器：`src/rl_sar/scripts/profile_lw_runtime_config.py`
- 清单生成器：`src/rl_sar/scripts/generate_lw_deployment_manifest.py`
- LW 实机启动文件：`src/rl_sar/launch/rl_real_LW.launch.py`
- LW 实机部署权威问题记录：`.learnings/LW_REAL_DEPLOYMENT_ISSUES.md`；按其中的
  Ordered Summary 和各问题 Resolution 查阅当前状态与证据
