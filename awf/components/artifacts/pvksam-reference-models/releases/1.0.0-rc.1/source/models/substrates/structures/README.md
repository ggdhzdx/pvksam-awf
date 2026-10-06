# 基底结构库

本目录只保存项目维护的、可追溯的基底母体结构。普通 SAMFlow 用户不直接选择这里的
文件；用户提交基底名称后，由 `substrates/*.toml` 选择已经验证的结构和配方。

## 目录生命周期

```text
substrates/structures/<substrate-key>/
├── sources/   # 从数据库或建模软件取得的原始文件，保持原字节不变
├── bulk/      # Bulk Seed，以及通过协议验收后登记的 Bulk Parent
├── slabs/     # 按晶面、终止方式和版本保存的无掺杂洁净预切片
└── manifest.json
```

三个层级不能混用：

- `sources/` 是证据，不允许原地修改；格式转换或对称标准化必须产生新文件。
- `bulk/` 中规范化得到的晶胞先登记为 `Bulk Seed`，不能直接声称为切面母体。只有使用
  下游同一势函数完成晶胞与原子位置松弛、并通过参考晶格和结构验收后，才能另存并登记
  为 `Bulk Parent`。Seed 永不被原地覆盖。
- `Bulk Parent` 才是自动切面的直接母体。必须记录来源、势函数哈希、组成、晶格、空间群、
  松弛合同、收敛指标和验收报告。
- `slabs/` 使用 `<hkl>/<termination>/` 分目录。每个预切片必须记录母体、切面参数、
  层数、实体厚度、真空、终止方式、原子数和哈希。
- Sn/F 等掺杂、目标超胞、吸附位点实例和具体 SAM 运行结构不放在这里；它们由基底
  配方确定性生成并进入不可变运行目录。

## 当前状态

- `ito/` 保存 Materials Project 的 Ia-3 In2O3、Materials Studio 的 I2_13 备选来源，
  已通过 `bulk_parent_relaxation_v1` 的 MACE Bulk Parent，以及从该母体按
  `surface_precut_v1` 重切并登记的三重复 In2O3(111) 非极性洁净预切片。原 Materials
  Studio 三层片保留为历史证据，但全原子几何对照不等价，不再作为目录配方输入。
- `fto/` 保存 Materials Studio 金红石 SnO2 来源和显式 6 原子 conventional bulk。
  已登记通过同一协议的 MACE Bulk Parent，以及五重复、桥氧完整、化学计量且形式非极性
  的 SnO2(110) 维护版松弛表面。未掺杂母表面的正式标定已登记一个稳定双齿磷酸位点
  原型和三个羧酸双齿原型，并保留被吸附弛豫拒绝的洁净几何组合。`substrates/fto.toml`
  已进一步登记单分子自适应近方形超胞、`F/(O+F)=3/40=7.5%` 默认值、五层允许 O 位、
  全局 4 Å F 间距、0–11% 用户覆盖范围和隐式替位施主电荷约定。FTO 致密单层覆盖度
  仍未登记，因此该模式明确拒绝而不会借用 ITO 参数。

每个已接受的 `bulk_parent` 都在 `bulk/records/` 固化计划、验收报告、运行清单、优化器
日志、逐步进度和实验原子位置检查。`runs/` 中的完整轨迹仍是运行产物；结构库记录足够
验证候选、合同、模型哈希和接受结论。早期运行验收报告没有原子位置门槛时，保留原报告
不改写，并增加带独立哈希的 `experimental_position_check` 记录补齐证据链。

## 实验结构验收

参考晶格和内部原子坐标必须来自同一份登记并校验哈希的实验 CIF；不能用 Materials
Project 或 Bulk Seed 的 DFT 坐标冒充实验坐标。当前 ITO 使用 Marezio (1966) 的单晶
X 射线 In2O3 结构（COD 2310009），FTO 的未掺杂母相使用 Bolzan 等 (1997) 的中子粉末
衍射 SnO2 结构（COD 2101853）。

内部坐标比较先将候选和实验结构按同一空间群标准化为常规晶胞，再搜索等价原点，并按
元素在周期性最小镜像距离下做一一匹配。位移距离使用实验晶格计算，因此不重复计入晶胞
长度偏差。报告同时保存总 RMS/最大位移、逐元素结果、逐 Wyckoff 位点结果和标准化后的
原子映射。`bulk_parent_relaxation_v1` 当前要求 RMS 不超过 0.05 Å、任一原子不超过
0.10 Å；晶格误差、空间群、组成、力和应力仍是相互独立的强制门槛。

各材料的 `manifest.json` 是本目录的机器可读索引。Git 路径不能替代 SHA-256；移动或
重命名不得改变来源文件哈希。

## 只读 Bulk Parent 松弛计划

维护者在运行昂贵计算前，先验证 Bulk Seed、模型身份、元素覆盖和验收合同。该命令只
输出 JSON，不启动 MACE、优化器或 GPU：

```bash
python scripts/substrate_library.py bulk-relaxation-plan \
  --substrate ITO \
  --model mace-mpa-0-medium.model \
  --model-element In \
  --model-element O
```

FTO 将元素声明替换为 `Sn` 和 `O`。`--model-element` 是维护者对模型覆盖范围的显式
声明，目前不冒充从模型文件自动解析出的证据。计划本身始终只读；
`execution_contract.executable=true` 只表示可以另行显式启动执行命令。

## 生成 Bulk Parent 候选

执行命令必须显式给出输出根目录，并要求可用的 MACE、PyTorch、ASE、spglib 和所选
设备。它用 `FixSymmetry + FrechetCellFilter + LBFGS` 同时松弛原子位置和允许的晶胞
自由度，不允许立方/四方晶胞在数值优化中降对称。规划器按“应力阈值 × 初始体积 ÷
力阈值”确定 cell-gradient 缩放，使 LBFGS 的统一 `fmax` 停止条件同时覆盖原子力和
晶胞应力，而不是在力收敛后留下超标残余应力：

```bash
python scripts/substrate_library.py bulk-relaxation-run \
  --substrate ITO \
  --model mace-mpa-0-medium.model \
  --model-element In \
  --model-element O \
  --output-root runs/bulk-relaxation \
  --device cuda
```

每次执行创建独立且不可覆盖的运行目录，保存计划快照、逐步进度、ASE 轨迹、优化器
日志、候选 CIF、验收报告和运行清单。只有力、应力、参考晶格、实验内部原子位置、晶胞
角、组成、原子数和空间群全部通过，候选才具备登记为 `Bulk Parent` 的资格；运行命令
本身不修改结构库清单，也不覆盖 Bulk Seed。

## 枚举和生成 Surface Precut

表面预切片从已接受的 Bulk Parent 出发，不能从 Bulk Seed 或历史片层反推。只读规划器
将常规晶胞 Miller 指数严格变换到 primitive cell，枚举一个周期内所有原子层间的切割
相位，并报告每个候选的组成、原子数、厚度、真空、顶/底原子层、形式电荷和形式偶极：

```bash
python scripts/substrate_library.py surface-precut-plan --substrate ITO
```

显式执行命令把全部候选写入新的不可覆盖目录，不直接修改结构库：

```bash
python scripts/substrate_library.py surface-precut-run \
  --substrate ITO \
  --output-root runs/surface-precut
```

ITO(111) 的 11 个切割区间中，只有零相位候选同时满足体相化学计量、形式电荷中性、
形式偶极阈值和 O 顶层约束。它含 120 个原子，实体厚度 8.0491 Å、真空 25 Å，并登记
最初登记为 `accepted_bulk_derived_unrelaxed_precut`；该几何预切片现已由正式松弛
维护表面取代，但仍保留为父子关系和初始几何证据。

历史三层片与新候选在统一面内尺度并做周期全原子匹配后，RMS 偏差为 0.4535 Å、最大
偏差为 0.7693 Å，超过 0.10 Å 的父子等价阈值。因此历史文件继续保留但不得再声明为
当前 Bulk Parent 的直接切片。

FTO 使用同一入口枚举 SnO2(110) 的三个切割区间。相位 0.5 的候选是唯一满足全部门槛
的上下对称桥氧终止，含 Sn10O20，表面晶胞为 6.8257 × 3.2297 Å，实体厚度
16.2929 Å，真空 25 Å。采用 (110) 和完整桥氧终止的依据是早期第一性原理表面研究；
五重复是面向后续吸附计算的保守初值。文献中的三层/五层弛豫差异小于 0.02 Å，但项目
仍要求用当前 MACE 势重新做厚度收敛，不能把文献收敛直接冒充本项目计算结果：

- [Rantala 等，SnO2(110) 表面弛豫](https://doi.org/10.1016/S0039-6028(98)00833-4)
- [环境相关的 SnO2 低指数面稳定性](https://doi.org/10.1063/1.3694033)

## 表面松弛与厚度收敛

### 同一晶面的暴露层筛选

正式登记松弛表面前，必须对指定 Miller 晶面的全部层间切割相位做一次完整筛选，不能
把 `preferred_top_elements = ["O"]` 当作能量结论。只读入口会列出每个切割相位的
顶层、底层、形式偶极和完整厚度序列：

```bash
python scripts/substrate_library.py surface-termination-screen-plan \
  --substrate ITO \
  --model mace-mpa-0-medium.model \
  --model-element In \
  --model-element O
```

显式执行入口会为每个切割相位生成分类证据，并对物理上可直接比较的候选松弛全部
厚度、生成暴露层选择报告：

```bash
python scripts/substrate_library.py surface-termination-screen-run \
  --substrate ITO \
  --model mace-mpa-0-medium.model \
  --model-element In \
  --model-element O \
  --output-root runs/surface-termination-screen \
  --device cuda
```

这里计算的 `gamma = (E_slab - N_formula * E_bulk_formula) / (2A)` 是一个 slab
上下两个表面的成对平均超额能。只有上下对称、形式非极性的情形才能直接解释为单个
终止面的表面能；上下不对称或带形式偶极的切片不能冒充某一个暴露层的独立表面能。
因此流程会登记它们的切割位置、上下表面组成和形式偶极，但不会对未补偿极性候选执行
没有明确物理含义的能量排名。只有化学计量、电中性、形式非极性、上下表面层组成一致、
力收敛且厚度收敛的候选才执行完整计算并进入最终能量排序。若未来要比较非化学计量或
极性终止，必须另建对称 slab 并声明元素化学势/补偿机制。

每次筛选完成后必须主动汇报，而不能只给出最终结构：总切层数、通过直接可比性门槛的
数量、实际弛豫数量、最终可排名数量和所有排除原因。若只有一个直接可比候选，必须明确
说明它是“唯一合格项”，不是在多候选能量竞争中胜出；若有多个最终可排名候选，必须
逐个列出收敛厚极限表面能，以及相对最低能候选高出的 `eV/Å²` 和 `J/m²`。这些字段由
`termination-selection-report.json` 的 `selection_summary` 强制保存。

ITO(111) 与 FTO(110) 的正式筛选输入、判据、逐厚度能量、排除原因、报告哈希和方法
修正经过已固化在
[2026-08-02 同晶面暴露层筛选正式验证记录](../../docs/surface-termination-screen-validation-2026-08-02.zh-CN.md)。
后续编写或执行完整计算流程时，应把该记录对应的步骤作为基底表面晋升前的必需阶段。

几何预切片登记后，维护者使用同一 `substrate_library.py` 进入下一道独立门槛。只读
规划器会验证：预切片和 Bulk Parent 的哈希链、MACE 模型是否与 Bulk Parent 完全一致、
模型元素覆盖、固定晶胞和双表面对称合同，以及清单声明的厚度序列。它只打印计划，
不会加载 MACE 或启动 GPU：

```bash
python scripts/substrate_library.py surface-relaxation-plan \
  --substrate ITO \
  --model mace-mpa-0-medium.model \
  --model-element In \
  --model-element O
```

显式执行时，每个厚度分别保存初始/松弛 CIF、轨迹、优化器日志、逐步进度和验收报告：

```bash
python scripts/substrate_library.py surface-relaxation-run \
  --substrate ITO \
  --model mace-mpa-0-medium.model \
  --model-element In \
  --model-element O \
  --output-root runs/surface-relaxation \
  --device cuda
```

当前 ITO 检查 3、4、5 个 oriented-unit repeats，FTO 检查 3、5、7 个 repeats。每个
slab 固定 Bulk Parent 给出的晶胞，只松弛原子位置，并用质心约束去除整体平移；上下
两个表面同时存在，因此表面能定义为
`gamma = (E_slab - N_formula * E_bulk_formula) / (2A)`。体相参考不是沿用旧日志里的
能量，而是在同一次运行中用同一个模型、精度、设备和计算器实例对已接受 Bulk Parent
重新做单点计算。最厚两个 slab 的表面能差不超过 0.001 eV/Å²（约 0.0160 J/m²）才通过
当前厚度收敛门槛。

筛选执行器本身不修改结构库。计算通过后，维护者使用只读
`surface-promotion-plan` 检查整条证据链，再通过显式 `surface-promotion-run` 创建新的
不可变维护包并更新结构清单。当前 ITO 3-repeat 与 FTO 5-repeat 松弛结构已经完成该
晋升，状态为 `accepted_relaxed_maintained_surface`；原几何预切片保留并标记为已取代。
