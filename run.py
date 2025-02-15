# -*- coding: utf-8 -*-
# @Time       : 2025/2/9 11:44
# @Author     : Marverlises
# @File       : run.py
# @Description: PyCharm

import logging
import os

from config import settings
from modules.spider_factory import SpiderFactory


def init_logger():
    if not os.path.exists(settings.LOGGING_FILE_PATH):
        os.makedirs(settings.LOGGING_FILE_PATH)
    log_file = os.path.join(settings.LOGGING_FILE_PATH, 'run.log')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=settings.LOGGING_LEVEL,
                        handlers=[
                            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                            logging.StreamHandler()
                        ])


if __name__ == '__main__':
    init_logger()
    spider_factory = SpiderFactory()
    # get all papers
    all_info = spider_factory.get_all_papers()
    # persist data
    spider_factory.persist_data()
