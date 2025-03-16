# -*- coding: utf-8 -*-
# @Time       : 2025/2/9 17:44
# @Author     : Marverlises
# @File       : spider_factory.py
# @Description: Factory for creating and managing paper spiders
import importlib
import logging
import os
from typing import List, Dict, Any, Optional, Union

from config import settings
from modules.paper_spider import PaperSpider
from modules.paper_sql import PaperSQL

logger = logging.getLogger(__name__)

class SpiderFactory:
    """
    Factory class for creating and managing paper spider instances.
    Handles spider creation, data retrieval, and persistence.
    """

    def __init__(self, 
                 conferences: List[str] = None,
                 years: List[str] = None,
                 db_instance: PaperSQL = None):
        """
        Initialize the spider factory.
        
        Args:
            conferences (List[str], optional): List of conference names. Defaults to settings.NEED_CONFERENCES_OR_JOURNALS.
            years (List[str], optional): List of years to fetch papers for. Defaults to settings.NEED_YEAR.
            db_instance (PaperSQL, optional): Database connection instance. If None, a new one is created.
        """
        self.conferences = conferences or settings.NEED_CONFERENCES_OR_JOURNALS
        self.years = years or settings.NEED_YEAR
        self.db = db_instance or PaperSQL()
        self.spiders = []
        self.initialize_spiders()
        
    def initialize_spiders(self) -> None:
        """
        Initialize all spider instances based on configured conferences and years.
        
        Returns:
            None
        """
        for conference in self.conferences:
            try:
                # Dynamically import the spider class
                module_path = f"modules.spiders.{conference.lower()}_paper_spider"
                class_name = f"{conference}PaperSpider"
                
                try:
                    module = importlib.import_module(module_path)
                    spider_class = getattr(module, class_name)
                    
                    # Create spider instance with shared DB connection
                    spider_instance = spider_class(year=self.years, paper_sql=self.db)
                    self.spiders.append(spider_instance)
                    logger.info(f"Successfully initialized spider for {conference}")
                    
                except (ImportError, AttributeError) as e:
                    logger.error(f"Failed to load spider for {conference}: {e}")
                    
            except Exception as e:
                logger.error(f"Unexpected error creating spider for {conference}: {e}")
                
        if not self.spiders:
            logger.warning("No spiders were initialized!")
    
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Fetch papers from all configured conferences.
        
        Returns:
            List[Dict[str, Any]]: Combined list of all papers from all conferences
        """
        all_papers = []
        for spider in self.spiders:
            try:
                papers = spider.fetch_paper_info()
                all_papers.extend(papers)
                logger.info(f"Fetched {len(papers)} papers from {spider.conference_name}")
            except Exception as e:
                logger.error(f"Error fetching papers from {spider.conference_name}: {e}")
                
        return all_papers
    
    def persist_all_papers(self) -> None:
        """
        Persist paper information from all spiders to the database.
        
        Returns:
            None
        """
        for spider in self.spiders:
            try:
                spider.persist_paper_info()
            except Exception as e:
                logger.error(f"Error persisting papers for {spider.conference_name}: {e}")
    
    def download_all_papers(self, output_dir: str = settings.PDF_SAVE_PATH) -> Dict[str, Dict[str, int]]:
        """
        Download papers from all conferences.
        
        Args:
            output_dir (str, optional): Directory to save papers to. Defaults to settings.PDF_SAVE_PATH.
            
        Returns:
            Dict[str, Dict[str, int]]: Statistics of downloaded papers by conference
        """
        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        download_stats = {}
        for spider in self.spiders:
            try:
                success_count, total_count = spider.download_papers(output_dir)
                download_stats[spider.conference_name] = {
                    'success': success_count,
                    'total': total_count,
                    'success_rate': round(success_count / total_count * 100, 2) if total_count > 0 else 0
                }
            except Exception as e:
                logger.error(f"Error downloading papers for {spider.conference_name}: {e}")
                download_stats[spider.conference_name] = {'success': 0, 'total': 0, 'success_rate': 0, 'error': str(e)}
                
        return download_stats
    
    def get_database_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Get statistics about papers stored in the database.
        
        Returns:
            Dict[str, Dict[str, int]]: Paper statistics by conference
        """
        stats = {}
        for conference in self.conferences:
            for year in self.years:
                table_name = f"{conference}_{year}"
                try:
                    # Get count of papers in the table
                    cursor = self.db.connection.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    
                    if table_name not in stats:
                        stats[table_name] = {'total_papers': 0}
                    
                    stats[table_name]['total_papers'] = count
                    
                except Exception as e:
                    logger.error(f"Error getting statistics for {table_name}: {e}")
                    stats[table_name] = {'total_papers': 0, 'error': str(e)}
        
        return stats
