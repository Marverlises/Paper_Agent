# -*- coding: utf-8 -*-
# @Time       : 2025/2/9 17:44
# @Author     : Marverlises
# @File       : spider_factory.py
# @Description: PyCharm
from config import settings
from modules.paper_sql import PaperSQL

from modules.spiders.acl_paper_spider import ACLPaperSpider
from modules.spiders.cvpr_paper_spider import CVPRPaperSpider
from modules.spiders.eccv_paper_spider import ECCVPaperSpider
from modules.spiders.ijcai_paper_spider import IJCAIPaperSpider


class SpiderFactory:
    titles = []
    abstracts = []
    spiders = []

    def __init__(self):
        self.set_paper_spiders()

    def set_paper_spiders(self, conferences=settings.NEED_CONFERENCES_OR_JOURNALS, year=settings.NEED_YEAR):
        """
        Factory function to return a list of PaperSpiders based on conference names.
        :param conferences:
        :param year:
        :return:
        """
        spiders = []
        for conference in conferences:
            if conference == "ACL":
                spiders.append(ACLPaperSpider(year))
            elif conference == "CVPR":
                spiders.append(CVPRPaperSpider(year))
            elif conference == "ECCV":
                spiders.append(ECCVPaperSpider(year))
            elif conference == "IJCAI":
                spiders.append(IJCAIPaperSpider(year))
            else:
                raise ValueError(f"Unsupported conference: {conference}")
        self.spiders = spiders
        return spiders

    def get_titles(self):
        """
        Get the titles of the papers from the database.
        """
        pass

    def get_abstracts(self):
        """
        Get the abstracts of the papers from the database.
        """
        pass

    def get_all_papers(self):
        """
        Get all the papers from the internet.
        """
        data = []
        for spider in self.spiders:
            data.extend(spider.fetch_paper_info())
        return data

    def persist_data(self):
        """
        Persist paper information to the database.
        """
        for spider in self.spiders:
            spider.persist_paper_info()
