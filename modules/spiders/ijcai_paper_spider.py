# -*- coding: utf-8 -*-
# @Time       : 2025/2/15 10:42
# @Author     : Marverlises
# @File       : ijcai_paper_spider.py
# @Description: Spider for IJCAI conference papers
import logging
from typing import List, Dict, Any

from config import settings
from modules.paper_spider import PaperSpider
from modules.paper_sql import PaperSQL

logger = logging.getLogger(__name__)

class IJCAIPaperSpider(PaperSpider, PaperSQL):
    """
    Spider for International Joint Conference on Artificial Intelligence (IJCAI) papers.
    """
    
    def __init__(self, year=settings.NEED_YEAR, paper_sql=None):
        """
        Initialize the IJCAI paper spider.
        
        Args:
            year (List[str]): Years to scrape papers for
            paper_sql: Database connection instance
        """
        super().__init__(year, paper_sql)
        PaperSQL.__init__(self)
        self.conference_name = "IJCAI"
        self.year_id_map = settings.YEAR_ID_MAP.get('IJCAI', {})
        
        if not self.year_id_map:
            logger.warning("No IJCAI conference IDs found in settings")
            
    def fetch_paper_info(self) -> List[Dict[str, Any]]:
        """
        Fetch all paper data from the IJCAI conference.
        
        Returns:
            List[Dict[str, Any]]: List of paper information dictionaries
        """
        # Get conference IDs for requested years
        need_conf_ids = []
        for year in self.years:
            if year in self.year_id_map:
                need_conf_ids.append((self.year_id_map[year], year))
            else:
                logger.warning(f"No conference ID found for IJCAI {year}")
        
        all_paper_data = []
        for conf_id, year in need_conf_ids:
            logger.info(f"Fetching IJCAI {year} papers with conference ID: {conf_id}")
            
            try:
                # Fetch papers using the utility method
                paper_data = self.util_fetch_publications(conf_id)
                
                if not paper_data:
                    logger.warning(f"No papers found for IJCAI {year}")
                    continue
                    
                # Process each paper
                for paper in paper_data:
                    # Extract the publication data
                    publication = paper.get('publication', {})
                    if not publication:
                        continue
                        
                    # Add year if missing
                    publication['year'] = year
                    
                    # Remove original id to avoid conflicts
                    if 'id' in publication:
                        publication.pop('id')
                        
                    # Rename abstract field if needed for consistency
                    if 'abstract' in publication and 'abstracts' not in publication:
                        publication['abstracts'] = publication['abstract']
                        
                    all_paper_data.append(publication)
                    
                logger.info(f"Fetched {len(paper_data)} papers from IJCAI {year}")
            except Exception as e:
                logger.error(f"Error fetching papers for IJCAI {year}: {e}")
                
        # Cache the paper data
        self.all_info = all_paper_data
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
