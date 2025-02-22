作业计划：基于 Semeval 2010 Task 8 的 BERT 关系抽取实验

1. 数据集
	•	采用 Semeval 2010 Task 8 数据集进行实验，该数据集用于关系分类任务，包含 9 种关系类型（+ 1 个 “Other” 类）。
	•	需要对数据进行预处理，以适配 BERT 模型的输入格式。

2. BERT 基础代码
	•	选用 BERT 预训练模型 作为关系分类的基准模型。
	•	主要使用 transformers 库（如 BERT-base-uncased），并进行 微调（fine-tuning）。

3. 运行基准测试
	•	在 Semeval 2010 Task 8 数据集上 运行基础 BERT 模型，记录 基准测试（baseline） 的性能指标（如 Accuracy、F1-score）。
	•	观察模型的训练过程和最终结果，分析其 在不同关系类别上的表现。

4. 改进

4.1 数据增强（可选）
	•	目标：扩充数据集，提升模型的泛化能力。
	•	方法：
	•	同义词替换（使用 WordNet、BERT MLM 生成替换词）
	•	回译（使用机器翻译进行数据扩展）
	•	伪标签（Pseudo-labeling）：利用已有模型生成新样本的标签

4.2 模型改进

4.2.1 引入 Named Entity Linking（NEL）
	•	目标：提升实体表示，使关系分类更加准确。
	•	方法：
	•	采用 SpaCy + Wikipedia 或 Wikidata 进行实体链接
	•	通过 预训练知识图谱嵌入（如 TransE） 提高实体表示
	•	结合 BERT + 实体特征，如使用 Concatenation 或 Attention 机制

4.2.2 微调模型/修改损失函数

1️⃣ 如何分析运行结果？
	•	分类报告（Classification Report）
	•	使用 sklearn.metrics.classification_report() 查看 Precision、Recall、F1-score。
	•	关注 “Other” 类别的占比，如果 F1-score 过高，可能影响其他类别。
	•	混淆矩阵（Confusion Matrix）
	•	观察 哪些类别容易被混淆，如果某些类别总被误分类，可以调整损失权重。
	•	Attention 分布
	•	观察 BERT 自注意力权重（Attention Weights），确定模型关注的关键部分。

2️⃣ 如何微调/优化损失函数？
	•	加权交叉熵（Weighted Cross-Entropy）
	•	如果 “Other” 类别样本过多，可以 降低其损失权重，提升其他类别的表现：