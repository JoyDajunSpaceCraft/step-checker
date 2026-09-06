# Step Checker / StepGap：交给 Codex 的代码修改任务

请在 `step-checker` 仓库启动 Codex，把本文件完整交给它。目标是把论文流程变成可审计、可训练、可复算的实现。

## 工作方式与证据基线

先读 AGENTS.md（若有）、git status、当前 commit、`revision_inputs/paper.pdf`、`revision_inputs/reviews.md`。独立分支修改，保留未提交改动与所有历史结果。完成代码和CPU测试；若已在用户GPU作业里，允许有限 smoke。不要自行启动付费API批量标注、完整RL矩阵、push或覆盖旧checkpoints。缺数据时继续完成独立代码，并列出精确需求。

审计基线 commit：`2a8d36fb69052e9657bc550d4fc5bb580658baa3`（2026-09-05）。实际文件主要为：

- `scripts/run_pipeline_v17.py`
- `build_sft_data/build_sft_data.py`
- `build_sft_data/build_hard_negative_aug.py`
- `train_checker/train_checker.py`

README 声称的一些目录/脚本并不在该 commit，尤其没有看到完整 GRPO 训练入口。不要根据README虚构可运行命令；先定位用户本地分支或其他项目中的真实训练代码。若仍缺失，建立明确接口和mock测试，标记 RL_BLOCKED，不能把SFT当成GRPO。

## P0：先解决论文与代码的版本矛盾

当前附加论文已经有typed P/R/F1、三seed结果、第二backbone、premise窗口与reward敏感性等附录。review比论文版本早；先建立表格→原始结果→commit/config/seed映射，验证已有结果，不能全部宣称缺失。

必须列入 `analysis/revision_evidence.csv` 的具体问题：

1. Figure2 图中“found quote + inference”直接no-gap，caption却说所有found quote都进入Stage D；caption还把无quote inference写成no-gap。§4/Appendix I的病例采用另一种路由。当前 `check_step` 有found-quote inference直接no-gap分支。这是定义差异，不是排版小问题。
2. Figure1图中step2是人物传记，caption却说film页面；step4图中no-gap，caption说IE。用同一真实trace自动生成示例图/表/caption，不能手写互相不一致的故事。
3. 本地代码标签 `unsupported_claim` 同时承接alignment、错误abstention与NLI contradiction；不能一律无损映射为论文的 Contradicted Claim。保留旧label和cause/path，建立版本化terminal语义以及映射表。
4. Appendix J/Table15仍含XX.X、XX和空case；K的retraction detector也有XX与[FILL]。它们不是已测结果，必须保留未完成状态，不能自动填合理数字。
5. Appendix S.2描述Llama单seed，而Table7 caption称两backbone三seed；核对原始runs。不存在的seed不应被生成。
6. 论文181 steps、107 gaps与README的confusion counts不完全对应；从冻结manifest重算，不选择最有利的版本。Appendix S.3以与human benchmark同trace相邻step筛hard negatives，需审计是否导致评估污染。

输出“历史v17行为”和“修订后规范”两个显式版本。优先定位产生论文主表的真实实现；若无证据确定，以配置开关实现备选路由、标记为NEW，不覆盖旧benchmark数字。不能为让论文成立而悄悄改变代码输出。

## P0：把checker契约与路由做成可测试模块

将step定义为带稳定ID的记录：question_id/trace_id/step_id、step_type、claim、query、local evidence、prior evidence、answer/action、gold（只用于评估）。区分query和可判真假的claim；Stage A的on-target不等于事实正确。

统一输出：schema_version、terminal、original_label、cause、pipeline_path、quote span及source id、NLI输入/输出、error/status、repair action、latency/API调用/token计数。API/NLI失败、解析失败、无quote、neutral必须可区分。

测试至少覆盖：off-target；正确/错误abstention；entity mismatch；found quote inference但不蕴含；结论的E/N/C；无local quote但prior支持；prior不支持；只有多片段联合才支持；只有query无claim；解析异常。外部模型响应使用fixtures/mocks测试控制流；另用真实小样本做语义验证，不能用mock通过来宣称checker准确率。

保留quote原文与字符位置，不用`[:150]`等截断破坏可验证span。比较短quote、句子/完整snippet和累计evidence是独立ablation。单snippet NLI不能识别需要联合证据的entailment；明确局限，若实现joint模式应单独命名和评测。

建议将routing schema集中在一个模块，并生成文档里的decision table。API request cache用模型版本/prompt/schema/输入hash作key；固定快照、重试与cost cap；不得把密钥写日志。

## P0：修复蒸馏数据与训练入口

1. `build_sft_data.py` 当前在展开step以后shuffle/split，可能让同题步骤落入训练和验证。改为先按question/trace分组切分，并检查跨数据集重复问句/trace；human test独立保留。输出split manifest、hash、来源和交叉污染检查。
2. `run_pipeline_v17.py` 的路由注释明确引用181-step benchmark的经验gap率。核对该集合是否参与规则选择；若参与，只能视为开发集，最终用未触碰数据评估。新建group split不能消除过去已发生的选择偏差。
3. 审计student学习的是LLM子阶段JSON还是整套checker终端输出。当前builder不等同于完整NLI决策树；不能声称student已蒸馏全部teacher路径。实现真实student推理adapter，接回固定NLI/router，保存teacher/student stage agreement。
4. `train_checker.py` 有硬样本weight字段，但collator返回时丢弃且Trainer未应用。要么实现并测试weighted loss，要么删除“2x weighting”声明；不能仅保留注释。
5. 用实际tokenizer chat template构造assistant-only loss mask，避免decode/re-encode估位置与找不到marker时整段loss。覆盖prompt包含marker、长输入截断、全-100 labels、空response；pad使用tokenizer.pad_token_id。
6. 区分论文Qwen2.5-7B、LoRA r=16与脚本默认Qwen3/r=32等配置。建立paper-config与exploratory-config；不要擅自换backbone以节省显存后继续称论文复现。
7. 首先支持单GPU LoRA。当前device_map=auto不是多卡DDP策略；只有明确实现分布式装载与采样后才能增卡。验证PyTorch/transformers/PEFT版本，BF16/FP16互斥路径、保存和加载adapter。当前TrainingArguments已经用`fp16=args.fp16 and not args.bf16`，不要虚报它同时开启两种精度。

## P0：补全能回应review的评估，而非仅增加训练量

建立 `scripts/revision/evaluate_checker.py`、`analyze_repairs.py`、`summarize_results.py`（建议新入口名，可按仓库风格调整）。

- 输出binary confusion matrix、positive prevalence、precision/recall/F1、balanced accuracy、MCC、macro-F1、各typed P/R/F1/support、Cohen kappa、question-level指标定义和分母。缺失预测、abstention、错误单列。
- Always-gap的`2g/(1+g)`与问题级`2w/(1+w)`是指定退化预测器的baseline/attainable value，**不是所有classifier的F1理论上界**；在定义适用时验证推导与样本计数。报告always-gap/no-gap baselines，修正文中ceiling措辞。
- 基于qid做paired bootstrap，不把同题steps当独立样本；RL跨seed均值/方差与evaluation bootstrap分开。只有82题/181steps时不能把大量重复API打分当新独立数据。
- 建立自动ablation配置，component removal是checker阶段屏蔽，不是删除policy trace中的步骤；为LLM-only与hybrid分别列出实际可移除组件。固定输入trace；报告stage removal对各类型而非仅F1影响。
- 修复行为分析使用冻结checker标注两策略，测CC后的真实retraction、IE/MB后的相关新search、最终grounding/answer。先定义互斥action规则、no-next-step/abstention/error分母；保留原始counts。新query不等于补到了缺证据。
- 单独用盲人工样本验证retraction与repair detector：口头“不是”而claim没变、表面改写、无效新搜索都要覆盖。人工标签缺失时输出annotation packet，不能把checker标签当人评。
- 按事先固定qid规则生成case，包含repair成功、EM正确但未grounded、checker false positive/negative；不是只挑成功案例。报告全部候选数量。

## P1：定位并集成GRPO，核对奖励归因

要求policy训练入口、retriever、checkpoint与数据manifest齐全后再执行。Student SFT和policy GRPO是不同阶段，不能训练完student就声称policy提升。

建立统一reward函数及mock rollout测试：endpoint EM + lambda*sum(base(type_i)+shape(action_i,type_{i-1}))，具体索引以论文/历史代码核对。特别检查最后一步奖励、截断、reset、no-gap边界和重复reward；记录actual token reward/advantage/mask。

正的CC检测奖励可能激励制造矛盾后自我纠正，不能仅凭“检测到了错误”解释。保留历史权重，单独实现zero/negative CC及shaping-off对照；测原始矛盾发生率、重复矛盾、retraction的真实性与最终support。不要把更多retraction自动解释为更可靠。

先寻找论文已有Search-only、Binary/Untyped-dense、Typed-base、Typed+Shape、straight-negative等原始结果。最小新的因果对照优先是：同轨迹输入/预算下typed与untyped reward、shaping开/关；与LLM-only reward比较属于P1（论文已明确没跑，不能从component ablation推导“只有本checker能作reward”）。固定独立开发集选checkpoint；test只最终评估。

## CRC执行与阶段gate

账号yuj49/account hdaqing；根目录 `/vast/hdaqing/yuj49`。模型/数据/结果/cache/env均可配置，移除活跃路径里的PSC hardcode；两仓库环境独立。保留Slurm CUDA_VISIBLE_DEVICES。初始1×RTX PRO6000 96GB；旧CUDA扩展不支持时用A10080/L40S smoke，不能假定显存更大就能运行旧软件。

| 阶段 | 最小工作 | 起步预算/资源 | 通过条件 |
|---|---|---|---|
| S0 | schema/routing/split/metrics/reward fixtures | CPU，不申请多GPU | 所有关键契约通过；历史与新版差异可追踪 |
| S1 | checker语义与API小样本 | 已授权API预算+CPU/1GPU NLI；先≤50steps | 无静默错误；span有效；实际成本/延迟已记录 |
| S2 | student单卡LoRA smoke | 1GPU，20–50 optimizer steps，先30–60min | 有assistant loss、无污染、adapter保存加载与推理可用 |
| S3 | 完整蒸馏/冻结checker evaluation | 单卡4h窗口起步，先估吞吐 | 固定配置、开发集与未触碰test；不以训练loss代表checker质量 |
| S4 | policy GRPO smoke | 训练链路找到后1×A10080/H200；OOM剖析后再2×A10080 | 完整group与reward attribution正确；无假装n=1的GRPO |
| S5 | 三seed最小policy矩阵 | 已有可验证结果复用；新增条件顺序跑 | 每seed固定budget；失败保存；独立评估可复算 |

论文S.2声称G=8、3000iterations，单H100约18h（Qwen）/22h（Llama）；这些是论文报告，尚未由本仓库验证，也不能直接换算成RTX/A100保证耗时。先测每update耗时/峰值显存，再决定24–48h申请。不能为了适配一张卡悄悄改变group size或有效batch。

## 必须交付

真实代码修改、必要测试、split和artifact manifest、版本化paper配置、CRC启动脚本、evidence ledger、含NOT_RUN状态的结果表、BLOCKERS。报告可复现命令与已执行验证。缺失GRPO时给出所需原始路径/文件列表与可工作的checker/SFT部分；不要为填论文数字凭空补实验结果。
