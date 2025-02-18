# -*- coding: utf-8 -*-
# @Time       : 2025/2/8 17:56
# @Author     : Marverlises
# @File       : settings.py
# @Description: PyCharm

# ------------------------------------
# 1. 基本配置
# ------------------------------------
import os

# 代理配置
os.environ['ALL_PROXY'] = 'http://127.0.0.1:7890'
# 输出日志的文件路径
LOG_FILE_PATH = 'logs/agent.log'
# 目前支持的会议
NEED_CONFERENCES_OR_JOURNALS = ['IJCAI']
# 需要获取的年份
NEED_YEAR = ['2024']
# 检索论文时的权重
SEARCH_WEIGHT = {'title': 0.15, 'abstract': 0.35, 'keywords': 0.5}
# 爬虫请求延迟，防止被封禁
DOWNLOAD_DELAY = 2
# 各个会议的年份对应的网站的ID
YEAR_ID_MAP = {'IJCAI': {'2024': '6582a84c261a3c46c0be364a', '2023': '64660edf10d52a2ee199e351',
                         '2022': '61a9949bdcb6d249681bd4f4'}}

# ------------------------------------
# 2. 数据库配置
# ------------------------------------

DATABASE_ENGINE = 'sqlite'
DB_SAVE_PATH = 'data/papers.db'  # 数据库保存路径

# ------------------------------------
# 3. PDF文件存储配置
# ------------------------------------

# PDF文件保存路径
PDF_SAVE_PATH = 'data/raw_papers/'

# 下载PDF时的最大重试次数
MAX_DOWNLOAD_RETRIES = 3

# ------------------------------------
# 4. 网络配置
# ------------------------------------

# 代理配置（如果需要使用代理）
USE_PROXY = False
PROXY = 'http://yourproxy.com:8080'  # 代理URL

# ------------------------------------
# 5. 数据分析配置
# ------------------------------------

# 分析报告输出路径
ANALYSIS_REPORT_PATH = 'data/analysis_results/'

# 分析的关键词数量（如在关键词统计中显示最常见的前10个）
KEYWORDS_COUNT = 10

# ------------------------------------
# 6. 用户配置 TODO
# ------------------------------------

# 用户默认偏好配置
USER_PREFERENCES = {
    'query_history': [],
    'preferred_format': 'pdf',  # 用户偏好的分析报告格式：'pdf' | 'excel'
    'enable_trend_analysis': True,
    'enable_keyword_analysis': True,
}

# ------------------------------------
# 7. 日志配置
# ------------------------------------

# 日志配置：日志级别可以是 DEBUG, INFO, WARNING, ERROR, CRITICAL
LOGGING_LEVEL = 'INFO'
LOGGING_FILE_PATH = './logs/'
