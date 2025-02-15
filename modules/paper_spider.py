# -*- coding: utf-8 -*-
# @Time       : 2025/2/8 18:01
# @Author     : Marverlises
# @File       : paper_spider.py
# @Description: PyCharm
import json
import logging
import sqlite3
import requests

from config import settings

logger = logging.getLogger(__name__)


class PaperSpider:
    def __init__(self, year=settings.NEED_YEAR):
        self.years = year

    def download_papers(self, titles):
        """
        Download papers based on the given list of titles.

        :param titles: list of paper titles
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @staticmethod
    def fetch_publications(conf_id):
        """
        获取会议文献数据并可选择存储到本地文件，使用日志记录请求过程。

        参数:
        - conf_id: 会议ID (必选)
        """
        # 设置请求的 URL
        url = "https://searchtest.aminer.cn/aminer-operation/web/conf/getWebPublications"

        # 设置请求头
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Authorization": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJhaWQiOiI2N2E4OGMwMDkxODljYjc0MzMyYTI1MDYiLCJhdWQiOlsiYTIyNzI5OWMtMGZhNC00Y2JlLTk1YTgtYThhZjdjZTQ4MzdjIl0sImNpZCI6ImEyMjcyOTljLTBmYTQtNGNiZS05NWE4LWE4YWY3Y2U0ODM3YyIsImV4cCI6MTczOTM1NTI2NiwiZ2VuZGVyIjowLCJpYXQiOjE3MzkzNTE2NjYsImlkIjoiNjdhODhjMDA5MTg5Y2I3NDMzMmEyNTA2IiwiaXNzIjoib2F1dGguYW1pbmVyLmNuIiwianRpIjoiOWUwNDk4NTQtMmNjOS00MWVmLWI5MGYtZDk3OGMyZDVmMGIxIiwibmJmIjoxNzM5MzUxNjY2LCJuaWNrbmFtZSI6Iis4NjEzMjg1NjM1ODA1Iiwic3ViIjoiNjdhODhjMDA5MTg5Y2I3NDMzMmEyNTA2IiwidCI6IlBhc3N3b3JkIn0.jQmfCKrT0uPCGi5M8Y7c-DVa46zmH72nPpqSYmOVvaroO7Rp7uf5QR9p1aTUxArBB3zpOCxBZ9YhDESNecsxqw9wR0vF_80EWHKJWaSh2MtMd6EJHoLK6D9BAF-rIQq3xl4JafQJKk5CyRZxGSF0I8MwjV-T8dT_yz6m69NRxta9St9k2bUyGFb2ORNNGN3BRWrpTfAher5YQGe-oquUmn3R3x9U20QRVyX5RGGQIeN3w5zgP2hwTh-AUbkfPItcAoAwnj0Tqyhve-LoAPLE9KkvNHoEtW_K6ZzwpICVuYfMzmi3tiwgqimh7zYfFwD_FyfpFbnjvosoWxSE9sFtEw",
            "Origin": "https://www.aminer.cn",
            "Referer": "https://www.aminer.cn/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "sec-ch-ua": "\"Not(A:Brand);v=\"99\", \"Google Chrome\";v=\"133\", \"Chromium\";v=\"133\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\""
        }

        # 请求体数据（JSON格式）
        data = {
            "fields": [
                "id", "year", "title", "title_zh", "abstract", "abstract_zh", "authors.id", "authors.name",
                "authors.name_zh", "keywords", "n_citation", "lang", "pdf", "ppt", "url", "resoureces"
            ],
            "confId": conf_id,
            "offset": 0,
            "size": 1000,
            "category": None,
            "sort": "view_num",
            "searchKeyword": ""  # 不使用搜索关键词
        }

        # 发送POST请求
        try:
            logger.info(f"Sending request to URL: {url} with confId: {conf_id}")
            response = requests.post(url, headers=headers, json=data)

            # 检查响应状态
            if response.status_code == 200:
                logger.info(f"Request successful for confId: {conf_id}")
                data = response.json()
            else:
                logger.error(f"Request failed with status code {response.status_code}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        return data['data']['records']

    def fetch_paper_info(self):
        """
        Fetch paper titles from the conference.
        """
        raise NotImplementedError("Subclasses must implement this method.")
