# -*- coding: utf-8 -*-
# @Time       : 2025/2/9 17:39
# @Author     : Marverlises
# @File       : cvpr_paper_spider.py
# @Description: PyCharm

from modules.paper_spider import PaperSpider


class CVPRPaperSpider(PaperSpider):
    def __init__(self, year):
        super().__init__(year)
        self.urls = "CVPR相关的URLs"  # 假设这里有需要的URL

    def scrape_paper_titles(self):
        """
        Scrape paper titles from the CVPR conference.
        """
        # 假设你已经实现了抓取代码
        pass

    def download_papers(self, titles):
        """
        Download papers based on the given list of titles.
        """
        # 假设你已经实现了下载代码
        pass
