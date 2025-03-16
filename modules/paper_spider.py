# -*- coding: utf-8 -*-
# @Time       : 2025/2/8 18:01
# @Author     : Marverlises
# @File       : paper_spider.py
# @Description: Base class for all paper spiders
import abc
import json
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple
from config import settings
from modules.paper_sql import PaperSQL

logger = logging.getLogger(__name__)

class PaperSpider(abc.ABC):
    """
    Abstract base class for all paper spiders.
    Defines the interface that all spiders must implement.
    """
    
    def __init__(self, year=settings.NEED_YEAR, paper_sql=None):
        """
        Initialize the paper spider.
        
        Args:
            year (List[str]): The year(s) to scrape papers from
            paper_sql (PaperSQL, optional): Database connection instance. If None, a new one is created.
        """
        self.years = year
        self.db = paper_sql if paper_sql else PaperSQL()
        self.all_info = []
        self.conference_name = self.__class__.__name__.replace('PaperSpider', '')
        
    @abc.abstractmethod
    def fetch_paper_info(self) -> List[Dict[str, Any]]:
        """
        Fetch all the paper data from the conference/journal.
        Must be implemented by all subclasses.
        
        Returns:
            List[Dict[str, Any]]: A list of paper information dictionaries
        """
        pass
    
    def persist_paper_info(self) -> None:
        """
        Persist paper information to the database.
        
        Returns:
            None
        """
        if not self.all_info:
            self.all_info = self.fetch_paper_info()
            
        if not self.all_info:
            logger.warning(f"No paper information to persist for {self.conference_name}")
            return
            
        table_name = f"{self.conference_name}_{'-'.join(self.years)}"
        columns = self.all_info[0].keys()
        
        # Create table if it doesn't exist
        self.db.create_table(table_name, columns)
        
        # Insert each paper
        for paper in self.all_info:
            self.db.insert_data(table_name, paper)
            
        logger.info(f"Successfully persisted {len(self.all_info)} papers for {self.conference_name} {'-'.join(self.years)}")
    
    def download_papers(self, output_dir: str = settings.PDF_SAVE_PATH) -> Tuple[int, int]:
        """
        Download papers to the specified directory.
        
        Args:
            output_dir (str): Directory to save papers to
            
        Returns:
            Tuple[int, int]: Number of successful downloads and total attempted downloads
        """
        from modules.utils import Utils
        
        if not self.all_info:
            self.all_info = self.fetch_paper_info()
            
        success_count = 0
        total_count = 0
        
        for paper in self.all_info:
            if 'pdf' in paper and paper['pdf']:
                total_count += 1
                pdf_url = paper['pdf']
                file_name = f"{self.conference_name}_{paper.get('year', 'unknown')}_{paper.get('id', str(total_count))}"
                
                if Utils.download_pdf_from_url(pdf_url, output_dir, file_name):
                    success_count += 1
                    
        logger.info(f"Downloaded {success_count}/{total_count} papers for {self.conference_name}")
        return success_count, total_count
    
    @staticmethod
    def util_fetch_publications(conf_id: str) -> List[Dict[str, Any]]:
        """
        Fetch publications from a conference using its ID.
        
        Args:
            conf_id (str): The conference ID
            
        Returns:
            List[Dict[str, Any]]: A list of publications
        """
        # Set the request URL
        url = "https://searchtest.aminer.cn/aminer-operation/web/conf/getWebPublications"

        # Set the request headers
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

        # Request data (JSON format)
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
            "searchKeyword": ""
        }

        # Send POST request
        try:
            logger.info(f"Sending request to URL: {url} with confId: {conf_id}")
            response = requests.post(url, headers=headers, json=data)

            # Check response status
            if response.status_code == 200:
                logger.info(f"Request successful for confId: {conf_id}")
                data = response.json()
                return data['data']['records']
            else:
                logger.error(f"Request failed with status code {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return []
