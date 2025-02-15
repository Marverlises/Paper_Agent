# -*- coding: utf-8 -*-
# @Time       : 2025/2/15 10:42
# @Author     : Marverlises
# @File       : eccv_paper_spider.py
# @Description: PyCharm
from config import settings
from modules.paper_spider import PaperSpider


class ECCVPaperSpider(PaperSpider):
    def __init__(self, year=settings.NEED_YEAR):
        super().__init__(year)

    def download_papers(self, titles):
        """
        Download papers based on the given list of titles.
        """
        # 假设你已经实现了下载代码
        pass

    def scrape_paper_info(self):
        """
        Scrape paper titles from the ECCV conference.
        """
        pass