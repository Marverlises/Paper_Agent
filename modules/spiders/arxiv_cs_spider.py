# -*- coding: utf-8 -*-
# @Time       : 2025/3/16 08:36
# @Author     : Marverlises
# @File       : arxiv_cs_spider.py
# @Description: Spider for ArXiv computer science papers

import logging
import re
import time
import urllib.parse
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

from config import settings
from modules.paper_spider import PaperSpider
from modules.utils import RequestUtils, Utils

logger = logging.getLogger(__name__)

class ArxivCSSpider(PaperSpider):
    """
    Spider for arXiv.org computer science papers.
    Supports searching by category, date, and keywords.
    """
    
    BASE_URL = "http://export.arxiv.org/api/query"
    CATEGORIES = [
        "cs.AI",   # Artificial Intelligence
        "cs.CL",   # Computation and Language (NLP)
        "cs.CV",   # Computer Vision
        "cs.LG",   # Machine Learning
        "cs.RO",   # Robotics
        "cs.IR",   # Information Retrieval
        "cs.HC",   # Human-Computer Interaction
        "cs.NE",   # Neural and Evolutionary Computing
    ]
    
    def __init__(self, 
                 year=settings.NEED_YEAR, 
                 paper_sql=None,
                 categories: Optional[List[str]] = None,
                 max_results: int = 100,
                 search_query: Optional[str] = None):
        """
        Initialize the ArXiv spider.
        
        Args:
            year (List[str]): Years to fetch papers for
            paper_sql: Database connection instance
            categories (List[str], optional): List of arXiv categories. Defaults to predefined CS categories.
            max_results (int, optional): Maximum number of results per query. Defaults to 100.
            search_query (str, optional): Additional search query to filter results. Defaults to None.
        """
        super().__init__(year, paper_sql)
        self.conference_name = "ArXiv"
        self.categories = categories or self.CATEGORIES
        self.max_results = max_results  # Max 2000 allowed by the API
        self.search_query = search_query
        
    def _construct_query(self, year: str, category: str) -> str:
        """
        Construct the query URL for arXiv API.
        
        Args:
            year (str): Year to fetch papers for
            category (str): arXiv category
            
        Returns:
            str: Query URL
        """
        # Set date range (whole year)
        start_date = f"{year}0101"
        
        # If this is the current year, use current date as end
        current_year = datetime.now().year
        if int(year) == current_year:
            end_date = datetime.now().strftime("%Y%m%d")
        else:
            end_date = f"{year}1231"
            
        # Construct query parameters
        params = {
            "search_query": f"cat:{category}+AND+submittedDate:[{start_date}+TO+{end_date}]",
            "start": 0,
            "max_results": self.max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        # Add additional search terms if provided
        if self.search_query:
            params["search_query"] += f"+AND+({self.search_query})"
            
        # Construct URL
        query_url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        
        return query_url
        
    def _parse_arxiv_response(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse the arXiv API response XML.
        
        Args:
            content (str): XML content from arXiv API
            
        Returns:
            List[Dict[str, Any]]: List of parsed paper records
        """
        papers = []
        
        try:
            # Parse XML response
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            root = ET.fromstring(content)
            
            # Process each entry (paper)
            for entry in root.findall('.//atom:entry', ns):
                try:
                    # Extract basic information
                    title = entry.find('./atom:title', ns).text.strip().replace('\n', ' ')
                    summary = entry.find('./atom:summary', ns).text.strip().replace('\n', ' ')
                    published = entry.find('./atom:published', ns).text
                    
                    # Get PDF link
                    pdf_url = None
                    for link in entry.findall('./atom:link', ns):
                        if link.get('title') == 'pdf':
                            pdf_url = link.get('href')
                            break
                            
                    # Get DOI if available
                    doi = None
                    doi_element = entry.find('.//arxiv:doi', ns)
                    if doi_element is not None:
                        doi = doi_element.text
                        
                    # Get authors
                    authors = []
                    for author in entry.findall('./atom:author', ns):
                        name = author.find('./atom:name', ns).text
                        authors.append({"name": name})
                        
                    # Get categories
                    categories = []
                    primary_category = entry.find('.//arxiv:primary_category', ns)
                    if primary_category is not None:
                        categories.append(primary_category.get('term'))
                        
                    for category in entry.findall('./atom:category', ns):
                        cat_term = category.get('term')
                        if cat_term not in categories:
                            categories.append(cat_term)
                            
                    # Extract year from published date
                    year = published[:4]
                    
                    # Get arXiv ID
                    id_url = entry.find('./atom:id', ns).text
                    arxiv_id = id_url.split('/')[-1] if id_url else None
                    
                    # Create paper record
                    paper = {
                        "title": title,
                        "year": year,
                        "authors": authors,
                        "abstracts": summary,
                        "arxiv_id": arxiv_id,
                        "doi": doi,
                        "published_date": published,
                        "categories": categories,
                        "url": id_url,
                        "pdf": pdf_url,
                        "source": "ArXiv"
                    }
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.error(f"Error parsing arXiv paper entry: {e}")
                    
            return papers
            
        except Exception as e:
            logger.error(f"Error parsing arXiv API response: {e}")
            return []
            
    def _fetch_papers_by_category_and_year(self, category: str, year: str) -> List[Dict[str, Any]]:
        """
        Fetch papers for a specific category and year.
        
        Args:
            category (str): arXiv category
            year (str): Year to fetch papers for
            
        Returns:
            List[Dict[str, Any]]: List of papers
        """
        papers = []
        
        try:
            query_url = self._construct_query(year, category)
            logger.info(f"Fetching arXiv papers for category {category}, year {year}")
            
            # Respect arXiv API rate limit (1 request per 3 seconds)
            time.sleep(3)
            
            response = requests.get(query_url)
            if response.status_code != 200:
                logger.error(f"ArXiv API returned status code {response.status_code}")
                return papers
                
            # Parse response
            papers = self._parse_arxiv_response(response.content)
            logger.info(f"Fetched {len(papers)} papers for arXiv category {category}, year {year}")
            
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching arXiv papers for category {category}, year {year}: {e}")
            return []
            
    def fetch_paper_info(self) -> List[Dict[str, Any]]:
        """
        Fetch all paper data from arXiv for the specified categories and years.
        
        Returns:
            List[Dict[str, Any]]: List of paper information dictionaries
        """
        all_papers = []
        
        for year in self.years:
            for category in self.categories:
                try:
                    category_papers = self._fetch_papers_by_category_and_year(category, year)
                    all_papers.extend(category_papers)
                    
                    # Sleep to respect API rate limits
                    time.sleep(3)
                    
                except Exception as e:
                    logger.error(f"Error fetching arXiv {category} papers for {year}: {e}")
                    
        # Remove duplicates by arXiv ID
        unique_papers = {}
        for paper in all_papers:
            arxiv_id = paper.get('arxiv_id')
            if arxiv_id and arxiv_id not in unique_papers:
                unique_papers[arxiv_id] = paper
                
        # Cache the paper data
        self.all_info = list(unique_papers.values())
        return self.all_info
        
    def search_arxiv(self, query: str, max_results: int = 50) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching a query.
        
        Args:
            query (str): Search query
            max_results (int, optional): Maximum number of results. Defaults to 50.
            
        Returns:
            List[Dict[str, Any]]: List of matching papers
        """
        try:
            # Construct query URL
            params = {
                "search_query": query,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            query_url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
            logger.info(f"Searching arXiv with query: {query}")
            
            # Send request
            response = requests.get(query_url)
            if response.status_code != 200:
                logger.error(f"ArXiv API returned status code {response.status_code}")
                return []
                
            # Parse response
            papers = self._parse_arxiv_response(response.content)
            logger.info(f"Found {len(papers)} papers matching query: {query}")
            
            return papers
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []