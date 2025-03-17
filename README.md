![img.png](https://raw.githubusercontent.com/Marverlises/PicGo/main/202503121016892.png)
# Paper Agent 🚀

## 快速挖掘顶会论文与个人兴趣匹配的研究

Paper Agent 是一个基于大模型的智能论文挖掘系统，旨在帮助研究人员、开发者及学生快速找到与自己兴趣相关的顶会论文。通过自动抓取各大顶级会议的最新论文，并根据用户的兴趣偏好进行精准匹配，Paper Agent 可以极大提高文献调研的效率。

---

### 🚀 主要功能

- **自动抓取顶会论文**：集成 ACL、CVPR、ECCV、IJCAI、ArXiv 等多个渠道的论文数据。
- **精准匹配**：通过大模型和语义搜索分析论文内容，智能匹配用户兴趣。
- **高效查阅**：支持从数据库中快速查询、获取感兴趣的论文，支持 PDF 下载。
- **深度分析**：利用大模型对论文进行深度解析，生成结构化报告。
- **持续更新**：随时更新数据库，确保获取最新的顶会论文。

---

### 📜 支持的会议/来源

Paper Agent 支持抓取以下会议/来源的论文：

- **IJCAI** (International Joint Conference on Artificial Intelligence)
- **CVPR** (Computer Vision and Pattern Recognition)
- **ECCV** (European Conference on Computer Vision)
- **ACL** (Association for Computational Linguistics)
- **ArXiv** (按类别抓取计算机科学领域论文)

更多会议支持正在开发中！🚀

---

### 💻 命令行工具

Paper Agent 包括以下主要功能：

```bash
# 抓取论文数据
python run.py fetch --conferences IJCAI CVPR --years 2023 2024

# 下载论文 PDF
python run.py download --conferences IJCAI --years 2024

# 搜索论文
python run.py search "Computer Vision" --method rag --top-k 5

# 处理论文（分析并生成报告）
python run.py process path/to/paper.pdf --api-key YOUR_API_KEY --model gpt-4

# 初始化 RAG 索引（用于语义搜索）
python run.py init-rag

# 列出可用的会议和年份
python run.py list-conferences
python run.py list-years --conference IJCAI
```

详细参数说明可通过 `python run.py -h` 或 `python run.py <command> -h` 查看。

---

### 🔧 安装与使用

#### 1. 克隆项目

```bash
git clone https://github.com/Marverlises/Paper_Agent.git
cd Paper_Agent
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 设置配置

编辑 `config/settings.py` 文件，根据需要调整参数：

```python
# 需要抓取的会议
NEED_CONFERENCES_OR_JOURNALS = ['IJCAI', 'CVPR']
# 需要获取的年份
NEED_YEAR = ['2024']
# 检索论文时的权重
SEARCH_WEIGHT = {'title': 0.15, 'abstracts': 0.35, 'keywords': 0.5}
```

#### 4. 运行

```bash
# 抓取论文
python run.py fetch

# 检索与分析论文
python run.py search "Reinforcement Learning"
python run.py process downloaded_papers/paper.pdf
```

---

### 📋 项目结构

```
Paper_Agent
├── LICENSE                # 项目的许可证信息
├── README.md              # 项目的介绍和使用指南
├── config                 # 配置文件目录
│   └── settings.py        # 项目配置文件，包含数据库连接、API 密钥等
├── data                   # 数据目录
│   ├── papers.db          # 存储论文数据的 SQLite 数据库
│   └── raw_papers/        # 下载的论文 PDF 文件存储位置
├── logs                   # 日志目录
│   └── run.log            # 程序运行日志
├── modules                # 模块目录
│   ├── paper_processor.py # 论文处理器，用于分析论文和生成报告
│   ├── paper_rag.py       # 基于检索增强生成的论文搜索实现
│   ├── paper_spider.py    # 论文爬虫的基类
│   ├── paper_sql.py       # 数据库操作模块
│   ├── llm_infer.py       # 大语言模型推理接口
│   ├── utils.py           # 通用工具函数
│   └── spiders/           # 不同会议的爬虫实现
│       ├── acl_paper_spider.py     # ACL 会议爬虫
│       ├── cvpr_paper_spider.py    # CVPR 会议爬虫
│       ├── eccv_paper_spider.py    # ECCV 会议爬虫
│       ├── ijcai_paper_spider.py   # IJCAI 会议爬虫
│       └── arxiv_cs_spider.py      # ArXiv 计算机科学领域爬虫
├── run.py                 # 主程序入口
└── requirements.txt       # 项目依赖列表
```

---

### 🔍 使用场景示例

1. **会议论文快速获取**

   ```bash
   # 抓取 CVPR 2024 的论文
   python run.py fetch --conferences CVPR --years 2024
   
   # 检查获取结果
   python run.py search "segmentation" --method db
   ```

2. **特定主题的论文调研**

   ```bash
   # 初始化语义检索索引
   python run.py init-rag
   
   # 使用语义搜索查找与特定主题相关的论文
   python run.py search "视觉-语言预训练模型的最新进展" --method rag --top-k 10
   ```

3. **论文深度分析**

   ```bash
   # 处理单篇论文并生成分析报告
   python run.py process path/to/paper.pdf --api-key YOUR_API_KEY
   ```

4. **批量下载与处理**

   ```bash
   # 下载论文
   python run.py download --conferences IJCAI --years 2024
   
   # 批量处理下载的论文
   python run.py process data/raw_papers/*.pdf --api-key YOUR_API_KEY
   ```

---

### 🚧 注意事项

- 使用论文分析功能需要提供 API 密钥，支持 OpenAI API 和兼容接口。
- 首次使用语义搜索前，需要运行 `python run.py init-rag` 初始化索引。
- 请遵守各论文数据源的使用规则和速率限制。
- 为避免使用时遇到问题，建议使用 Python 3.8 或更高版本。

---

### 🔮 未来计划

- 添加更多会议和期刊的支持
- 实现基于用户兴趣的自动推荐功能
- 提供 Web 界面，使操作更加便捷
- 支持更多大语言模型，提供更丰富的分析功能

---

### 📄 许可证

本项目采用 Apache License Version 2.0 许可证。详情请参见 [LICENSE](LICENSE) 文件。

