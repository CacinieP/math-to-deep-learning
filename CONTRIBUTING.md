# 贡献指南

感谢你对 **math-to-deep-learning** 的兴趣！本仓库的目标是用应用数学串联数学知识与深度学习。欢迎贡献。

## 如何贡献

### 报告问题

发现错误、断链、表述不清？请开 Issue，说明：
- 涉及的文件路径
- 问题描述
- 建议的修改

### 修正笔误与排版

小修复（错字、公式排版、链接失效）直接提 PR，无需先开 Issue。

### 贡献新内容

新文章请遵循仓库的**三层叙事**模板：

1. **纯数学层**：公式推导，带 `[[Mathematics-Universe/...]]` 双向链接指向母仓库
2. **应用映射层**：把数学概念映射到 ML，给图示 + 类比 + 直觉
3. **工程实现层**：最小可运行的 PyTorch 代码示例

每篇应包含：
- 篇首难度标注（`[基础]` / `[标准]` / `[进阶]` / `[前沿]`）
- 一句话点睛（`> ...`）
- 完整映射图（ASCII 三层框图）
- 直觉总结
- 延伸阅读（论文 + 关联文章）
- 联系网络框（`⬆上游 ⬇下游 ↔横联 🔗跨域`）

## 写作规范

- **公式**：用 `$...$` / `$$...$$`，在线阅读版使用 MathJax
- **代码**：标注语言（```python），保证可运行（至少 import + 几行）
- **链接**：内部链接用相对路径 `[[PART-02/01-神经网络基础]]`，跨仓库用 `[[Mathematics-Universe/03-高等数学/...]]`
- **术语**：中文为主，专有名词首次出现给英文（如"反向传播（backpropagation）"）
- **难度递进**：从纯数学直觉 → ML 映射 → 工程实现，层层深入

## 协作流程

1. Fork 仓库
2. 开分支：`git checkout -b feat/your-topic`
3. 提交，commit message 用中文描述
4. 开 PR，说明改动内容与动机

PR 合并前会检查：
- [ ] 遵循三层叙事模板
- [ ] 公式正确、代码可运行
- [ ] 双向链接有效
- [ ] 联系网络框完整

## 许可证

贡献内容遵循 [CC-BY-SA-4.0](./LICENSE)，与仓库整体一致。

---

> 数学是深度学习的母语，代码是它的语法。欢迎一起把这条"从纸笔推导到 GPU 训练"的理解线织得更密。


## 在线阅读版与本地检查

在线版从仓库现有 Markdown 生成，源文件仍是唯一内容来源。普通引用优先使用标准 Markdown 相对链接；跨仓库链接使用 GitHub 的 `blob/main` 文件地址，构建时会转换成配套在线读本的地址。兼容已有 `[[文件名]]`，但目标必须存在且唯一。代码块中的链接示意保持原样。

先将两个仓库放在相邻目录，使用 Python 3.12+ 和 Node.js 22：

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements-site.txt
npm ci --ignore-scripts
python -m unittest discover -s scripts -p 'test_site.py'
python scripts/prepare_site.py --peer ../Mathematics-Universe
python -m mkdocs build --strict
python scripts/check_site.py
node scripts/check_math.cjs
python -m mkdocs serve
```

编辑 Markdown 后需重新运行 `prepare_site.py`。`.site-docs/`、`site/` 和 `node_modules/` 都是生成文件，不提交。数学资源随站点部署，不依赖运行时 CDN。

PR 会运行构建和链接检查；合并到 `main` 后，GitHub Actions 发布到 GitHub Pages。仓库 Pages 的 Source 应设置为 **GitHub Actions**。

## Python 示例回归

新增或修改完整代码定义后，运行：

```sh
pip install -r requirements-examples.txt
python scripts/test_examples.py
```

测试从正文读取函数/类，执行小规模 CPU 数值检查；接口片段须注明所需上下文。CI 将示例回归与阅读站构建分开运行，两者均通过才部署。


## GitHub Wiki 同步

Wiki 与在线阅读版都从同一批 Markdown 生成。先在相邻目录分别完成两个站点的准备、严格构建和检查，再运行：

```sh
python -m unittest discover -s scripts -p 'test_wiki.py'
python scripts/prepare_wiki.py --peer ../PEER
```

将 `PEER` 替换为配套仓库目录。输出在 `.wiki-docs/`（不提交主仓库）；包含全文页面、章节导航、页脚及校验清单。发布前须先通过内容审核、公式与链接检查并合并源内容；重新生成 Wiki，使页脚标记正确的源提交。

首次发布须在 GitHub Wiki 网页创建首页，再克隆 `https://github.com/CacinieP/REPO.wiki.git`。将生成的 Markdown 同步到该克隆，检查差异后提交并推送其默认分支。保留 Wiki 自身 `.git` 与历史；校验清单 `.manifest.json` 用于核对页面哈希，不作为正文发布。不要直接维护两套不同正文。

链接与锚点按 Wiki 路由转换，代码示例保持原样。所有页面保留在线阅读入口与 CC BY-SA 4.0 归属信息。修改内容后需要同步两个发布渠道。
