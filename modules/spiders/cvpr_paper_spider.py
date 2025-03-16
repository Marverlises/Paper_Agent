# -*- coding: utf-8 -*-
# @Time       : 2025/2/15 10:14
# @Author     : Marverlises
# @File       : cvpr_paper_spider.py
# @Description: Spider for CVPR conference papers
import logging
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup

from config import settings
from modules.paper_spider import PaperSpider
from modules.utils import RequestUtils

logger = logging.getLogger(__name__)

class CVPRPaperSpider(PaperSpider):
    """
    Spider for Computer Vision and Pattern Recognition (CVPR) conference papers.
    """
    
    BASE_URL = "https://openaccess.thecvf.com/CVPR{year}"
    PAPER_URL = "https://openaccess.thecvf.com/"
    
    def __init__(self, year=settings.NEED_YEAR, paper_sql=None):
        """
        Initialize the CVPR paper spider.
        
        Args:
            year (List[str]): Years to scrape papers for
            paper_sql: Database connection instance
        """
        super().__init__(year, paper_sql)
        self.conference_name = "CVPR"
        
    def _fetch_papers_for_year(self, year: str) -> List[Dict[str, Any]]:
        """
        Fetch papers for a specific year.
        
        Args:
            year (str): Year to fetch papers for
            
        Returns:
            List[Dict[str, Any]]: List of papers for the year
        """
        papers = []
        
        # Build URL for the year
        url = self.BASE_URL.format(year=year)
        logger.info(f"Fetching CVPR {year} papers from {url}")
        
        try:
            # Get main conference page
            response = RequestUtils.make_request(url)
            if not response:
                logger.error(f"Failed to fetch CVPR {year} main page")
                return papers
                
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find links to paper pages
            paper_links = []
            for link in soup.find_all('a', href=True):
                if '/content_' in link['href'] and 'day=' in link['href']:
                    paper_links.append(link['href'])
                    
            logger.info(f"Found {len(paper_links)} paper links for CVPR {year}")
            
            # Process each paper link to get papers
            for link in paper_links:
                paper_list_url = self.PAPER_URL + link
                paper_response = RequestUtils.make_request(paper_list_url)
                
                if not paper_response:
                    logger.warning(f"Failed to fetch paper list from {paper_list_url}")
                    continue
                    
                paper_soup = BeautifulSoup(paper_response.text, 'html.parser')
                paper_items = paper_soup.find_all('dt', {'class': 'ptitle'})
                
                # Process each paper item
                for paper_item in paper_items:
                    try:
                        # Get title
                        title = paper_item.text.strip()
                        
                        # Get authors
                        authors_item = paper_item.find_next('dd')
                        author_text = authors_item.text.strip() if authors_item else ""
                        author_list = [{"name": name.strip()} for name in author_text.split(',') if name.strip()]
                        
                        # Get paper URL if available
                        paper_link = paper_item.find('a', href=True)
                        paper_url = self.PAPER_URL + paper_link['href'] if paper_link else ""
                        
                        # Get PDF URL if available
                        pdf_url = ""
                        if paper_link:
                            detail_url = self.PAPER_URL + paper_link['href']
                            detail_response = RequestUtils.make_request(detail_url)
                            
                            if detail_response:
                                detail_soup = BeautifulSoup(detail_response.text, 'html.parser')
                                pdf_link = detail_soup.find('a', {'class': 'btn', 'href': lambda x: x and x.endswith('.pdf')})
                                if pdf_link:
                                    pdf_url = self.PAPER_URL + pdf_link['href']
                                    
                                # Try to get abstract
                                abstract = ""
                                abstract_div = detail_soup.find('div', {'id': 'abstract'})
                                if abstract_div:
                                    abstract = abstract_div.text.strip()
                        
                        # Create paper record
                        paper = {
                            "title": title,
                            "year": year,
                            "authors": author_list,
                            "abstracts": abstract,
                            "url": paper_url,
                            "pdf": pdf_url,
                            "conference": "CVPR"
                        }
                        
                        papers.append(paper)
                        
                    except Exception as e:
                        logger.error(f"Error processing paper in CVPR {year}: {e}")
                        
            logger.info(f"Fetched {len(papers)} papers for CVPR {year}")
            
        except Exception as e:
            logger.error(f"Error fetching CVPR {year} papers: {e}")
            
        return papers
        
    def fetch_paper_info(self) -> List[Dict[str, Any]]:
        """
        Fetch all paper data from CVPR conferences.
        
        Returns:
            List[Dict[str, Any]]: List of paper information dictionaries
        """
        all_papers = []
        
        for year in self.years:
            try:
                year_papers = self._fetch_papers_for_year(year)
                all_papers.extend(year_papers)
            except Exception as e:
                logger.error(f"Error fetching CVPR {year} papers: {e}")
                
        # Cache the paper data
        self.all_info = all_papers
        return all_papers
