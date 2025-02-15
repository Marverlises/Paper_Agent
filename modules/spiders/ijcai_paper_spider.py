# -*- coding: utf-8 -*-
# @Time       : 2025/2/15 10:42
# @Author     : Marverlises
# @File       : ijcai_paper_spider.py
# @Description: PyCharm

from config import settings
from modules.paper_spider import PaperSpider
from modules.paper_sql import PaperSQL


class IJCAIPaperSpider(PaperSpider, PaperSQL):
    def __init__(self, year=settings.NEED_YEAR):
        super().__init__(year)
        PaperSQL.__init__(self)
        self.year_id_map = settings.YEAR_ID_MAP['IJCAI']
        self.all_info = self.fetch_paper_info()

    def fetch_paper_info(self):
        """
        Fetch paper titles from the IJCAI conference.
        """
        need_conf_id = [(id, year) for year, id in self.year_id_map.items() if year in settings.NEED_YEAR]
        all_paper_data = []
        for item in need_conf_id:
            conference_id = item[0]
            year = item[1]
            paper_data = self.fetch_publications(conference_id)
            for paper in paper_data:
                paper['publication']['year'] = year
                # remove original id
                paper['publication'].pop('id')
                all_paper_data.append(paper['publication'])
        return all_paper_data

    def persist_paper_info(self):
        """
        Persist paper information to the database.
        """
        table_name = f"IJCAI_{'-'.join(settings.NEED_YEAR)}"
        columns = self.all_info[0].keys()
        self.create_table(table_name, columns)

        for paper in self.all_info:
            self.insert_data(table_name, paper)
