![img.png](img.png)
# Paper Agent 🚀

## 快速挖掘顶会论文与个人兴趣匹配的研究

Paper Agent 是一个基于大模型的智能论文挖掘系统，旨在帮助研究人员、开发者及学生快速找到与自己兴趣相关的顶会论文。通过自动抓取各大顶级会议的最新论文，并根据用户的兴趣偏好进行精准匹配，Paper Agent 可以极大提高文献调研的效率。

---

### 🚀 主要功能

- **自动抓取顶会论文**：集成 ACL、CVPR、ECCV、IJCAI 等多个顶级会议的论文数据。
- **精准匹配**：通过大模型分析论文内容，智能匹配用户兴趣。
- **高效查阅**：支持从数据库中快速查询、获取感兴趣的论文，支持 PDF 下载。
- **持续更新**：随时更新数据库，确保获取最新的顶会论文。

---

### 📜 支持的会议

Paper Agent 支持抓取以下会议的论文：

- **ACL** (Association for Computational Linguistics)
- **CVPR** (Computer Vision and Pattern Recognition)
- **ECCV** (European Conference on Computer Vision)
- **IJCAI** (International Joint Conference on Artificial Intelligence)

更多会议支持正在开发中！🚀

---

### 💡 项目目标

借助最新的自然语言处理技术和深度学习模型，自动化从顶会获取论文，并通过智能匹配用户兴趣，帮助研究者节省大量查找时间，提升科研效率。

---

### 🔧 安装与使用

#### 1. 克隆项目

```bash
git clone https://github.com/Marverlises/Paper_Agent.git
cd Paper_Agent
pip install -r requirements.txt
```

项目文件树

```
Paper_Agent
├── LICENSE                # 项目的许可证信息，说明版权和使用许可
├── README.md              # 项目的介绍、安装和使用指南
├── config                 # 配置文件目录
│   └── settings.py        # 项目配置文件，包含数据库连接、API 密钥、全局设置等配置
├── data                   # 数据目录
│   └── papers.db          # 存储抓取的论文数据的 SQLite 数据库文件
├── logs                   # 日志目录
│   └── run.log            # 存储程序运行日志，记录错误、警告及其他运行时信息
├── modules                # 模块目录
│   ├── paper_spider.py    # 负责抓取论文数据的爬虫脚本
│   ├── paper_sql.py       # 用于与数据库进行交互的脚本，包括插入、查询等功能
│   ├── spider_factory.py  # 创建不同论文爬虫实例的工厂类，允许根据需要选择合适的爬虫
│   │   ├── acl_paper_spider.py  # 用于抓取 ACL (Association for Computational Linguistics) 会议的论文数据
│   │   ├── cvpr_paper_spider.py # 用于抓取 CVPR (Computer Vision and Pattern Recognition) 会议的论文数据
│   │   ├── eccv_paper_spider.py # 用于抓取 ECCV (European Conference on Computer Vision) 会议的论文数据
│   │   └── ijcai_paper_spider.py # 用于抓取 IJCAI (International Joint Conference on Artificial Intelligence) 会议的论文数据
│   └── utils.py           # 存放项目中常用的工具函数，比如数据处理、日志管理等
├── run.py                 # 项目的入口脚本，启动程序并控制爬虫抓取流程
└── tests/                 # 测试目录

```

