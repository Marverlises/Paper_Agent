# -*- coding: utf-8 -*-
# @Time       : 2025/2/9 11:29
# @Author     : Marverlises
# @File       : acl_paper_spider.py
# @Description: PyCharm
from config import settings
from modules.paper_spider import PaperSpider


class ACLPaperSpider(PaperSpider):
    def __init__(self, year=settings.NEED_YEAR):
        super().__init__(year)

    def download_papers(self, titles):
        """
        Download papers based on the given list of titles.
        """
        pass
