#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time       : 2025/2/15 18:30
# @Author     : Marverlises
# @File       : run.py
# @Description: Main entry point for the Paper Agent

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

from config import settings
from modules.spider_factory import SpiderFactory
from modules.paper_processor import PaperProcessor
from modules.paper_sql import PaperSQL
from modules.paper_rag import RAGRetrieval

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'run.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def ensure_directories():
    """
    Ensure all necessary directories exist.
    """
    directories = [
        'logs',
        'data',
        'data/raw_papers',
        settings.ANALYSIS_REPORT_PATH,
        os.path.dirname(settings.DB_SAVE_PATH)
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    logger.info("Directory structure verified")

def fetch_papers(conferences: List[str], years: List[str]) -> bool:
    """
    Fetch papers from specified conferences and years.
    
    Args:
        conferences (List[str]): List of conference names
        years (List[str]): List of years
        
    Returns:
        bool: Success status
    """
    try:
        logger.info(f"Fetching papers for conferences: {conferences}, years: {years}")
        
        # Create spider factory
        factory = SpiderFactory(conferences=conferences, years=years)
        
        # Fetch and persist papers
        papers = factory.get_all_papers()
        logger.info(f"Fetched {len(papers)} papers in total")
        
        factory.persist_all_papers()
        logger.info("Papers persisted to database")
        
        return True
    except Exception as e:
        logger.error(f"Error fetching papers: {e}")
        return False

def download_papers(conferences: List[str], years: List[str], output_dir: Optional[str] = None) -> Dict[str, Dict[str, int]]:
    """
    Download paper PDFs from specified conferences and years.
    
    Args:
        conferences (List[str]): List of conference names
        years (List[str]): List of years
        output_dir (Optional[str]): Output directory for PDFs
        
    Returns:
        Dict[str, Dict[str, int]]: Download statistics by conference
    """
    try:
        logger.info(f"Downloading papers for conferences: {conferences}, years: {years}")
        
        # Create spider factory
        factory = SpiderFactory(conferences=conferences, years=years)
        
        # Download papers
        stats = factory.download_all_papers(output_dir or settings.PDF_SAVE_PATH)
        
        # Print summary
        total_success = sum(s['success'] for s in stats.values())
        total_papers = sum(s['total'] for s in stats.values())
        
        logger.info(f"Downloaded {total_success}/{total_papers} papers ({total_success/total_papers*100:.2f}% success rate)")
        
        return stats
    except Exception as e:
        logger.error(f"Error downloading papers: {e}")
        return {}

def search_papers(query: str, top_k: int = 5, method: str = 'rag', search_method: str = 'weighted') -> List[Dict[str, Any]]:
    """
    Search for papers matching a query.
    
    Args:
        query (str): Search query
        top_k (int): Number of results to return
        method (str): Search method ('rag' for semantic search, 'db' for database search)
        search_method (str): Search method for RAG ('weighted' or 'title')
        
    Returns:
        List[Dict[str, Any]]: Matching papers
    """
    try:
        if method.lower() == 'rag':
            # Use RAG for semantic search
            logger.info(f"Performing RAG search for: {query}")
            
            # Initialize RAG
            rag = RAGRetrieval(
                db_path=settings.DB_SAVE_PATH,
                faiss_index_path=os.path.join('data', 'faiss_index.index'),
                search_weight=settings.SEARCH_WEIGHT
            )
            
            # Load model
            rag.load_model(settings.RAG_MODEL_PATH)
            
            # Get all existing tables
            with PaperSQL(settings.DB_SAVE_PATH) as db:
                cursor = db.connection.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
            results = []
            
            # Search in each table
            for table in tables:
                try:
                    rag.load_info_from_db(table)
                    rag.compute_embeddings()
                    rag.create_faiss_index()
                    
                    similar_papers = rag.get_most_similar_papers(query, top_k=top_k, method=search_method)
                    
                    # Add table info to results
                    for i, paper_info in similar_papers.items():
                        paper_info['table'] = table
                        results.append(paper_info)
                        
                except Exception as e:
                    logger.error(f"Error searching in table {table}: {e}")
                    
            # Sort results by distance
            results.sort(key=lambda x: x.get('distance', float('inf')))
            
            # Return top_k results
            return results[:top_k]
        else:
            # Use database search
            logger.info(f"Performing database search for: {query}")
            
            with PaperSQL(settings.DB_SAVE_PATH) as db:
                results = db.search_papers(query, limit=top_k)
                
            return results
    except Exception as e:
        logger.error(f"Error searching papers: {e}")
        return []

def process_papers(pdf_paths: List[str], 
                  api_key: Optional[str] = None, 
                  base_url: Optional[str] = None,
                  model_name: Optional[str] = None,
                  output_format: str = 'pdf') -> List[str]:
    """
    Process papers and generate reports.
    
    Args:
        pdf_paths (List[str]): List of PDF file paths
        api_key (Optional[str]): API key for LLM service
        base_url (Optional[str]): Base URL for LLM service
        model_name (Optional[str]): Model name for LLM service
        output_format (str): Output format ('pdf' or 'md')
        
    Returns:
        List[str]: List of report file paths
    """
    try:
        # Use settings if parameters not provided
        api_key = api_key or os.environ.get("LLM_API_KEY")
        base_url = base_url or os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        model_name = model_name or os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        
        if not api_key:
            logger.error("No API key provided for LLM service")
            return []
            
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(settings.ANALYSIS_REPORT_PATH, f"reports_{timestamp}")
        
        # Initialize processor
        processor = PaperProcessor(
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
            output_dir=output_dir
        )
        
        # Get metadata for each paper
        metadata_list = []
        for pdf_path in pdf_paths:
            paper_name = os.path.basename(pdf_path).split('.')[0]
            
            # Try to find paper metadata in database
            found_metadata = None
            
            with PaperSQL(settings.DB_SAVE_PATH) as db:
                # Search by filename pattern
                results = db.search_papers(paper_name)
                
                if results:
                    # Use the first result as metadata
                    found_metadata = results[0]
                    
            metadata_list.append(found_metadata or {"title": paper_name})
            
        # Process papers with metadata
        report_paths = processor.batch_process_papers(
            pdf_paths=pdf_paths,
            metadata_list=metadata_list,
            output_format=output_format
        )
        
        logger.info(f"Generated {len(report_paths)} reports in {output_dir}")
        
        return report_paths
    except Exception as e:
        logger.error(f"Error processing papers: {e}")
        return []

def init_rag_index(force_rebuild: bool = False) -> bool:
    """
    Initialize or rebuild the RAG index.
    
    Args:
        force_rebuild (bool): Force rebuild the index even if it exists
        
    Returns:
        bool: Success status
    """
    try:
        logger.info("Initializing RAG index")
        
        # Check if index already exists
        index_path = os.path.join('data', 'faiss_index.index')
        index_files = [
            index_path + "_title",
            index_path + "_abstract",
            index_path + "_keywords",
            index_path + "_weighted"
        ]
        
        if all(os.path.exists(f) for f in index_files) and not force_rebuild:
            logger.info("RAG index already exists. Use --force-rebuild to rebuild.")
            return True
            
        # Initialize RAG
        rag = RAGRetrieval(
            db_path=settings.DB_SAVE_PATH,
            faiss_index_path=index_path,
            search_weight=settings.SEARCH_WEIGHT
        )
        
        # Load model
        rag.load_model(settings.RAG_MODEL_PATH)
        
        # Get all existing tables
        with PaperSQL(settings.DB_SAVE_PATH) as db:
            cursor = db.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
        # Index each table
        for table in tables:
            try:
                logger.info(f"Indexing table: {table}")
                rag.load_info_from_db(table)
                rag.compute_embeddings()
                rag.create_faiss_index()
                rag.persist_faiss_index()
            except Exception as e:
                logger.error(f"Error indexing table {table}: {e}")
                
        logger.info("RAG index initialized successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing RAG index: {e}")
        return False

def list_conferences() -> List[str]:
    """
    List all available conferences.
    
    Returns:
        List[str]: List of conference names
    """
    return list(settings.YEAR_ID_MAP.keys())

def list_years(conference: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List all available years for a conference.
    
    Args:
        conference (Optional[str]): Conference name. If None, returns years for all conferences.
        
    Returns:
        Dict[str, List[str]]: Dictionary mapping conference names to lists of years
    """
    result = {}
    
    if conference:
        if conference in settings.YEAR_ID_MAP:
            result[conference] = list(settings.YEAR_ID_MAP[conference].keys())
    else:
        for conf, years in settings.YEAR_ID_MAP.items():
            result[conf] = list(years.keys())
            
    return result

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Paper Agent - Tool for analyzing academic papers")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Fetch papers command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch papers from conferences")
    fetch_parser.add_argument("--conferences", "-c", nargs="+", default=settings.NEED_CONFERENCES_OR_JOURNALS, 
                             help="List of conferences to fetch papers from")
    fetch_parser.add_argument("--years", "-y", nargs="+", default=settings.NEED_YEAR,
                             help="List of years to fetch papers for")
    
    # Download papers command
    download_parser = subparsers.add_parser("download", help="Download paper PDFs")
    download_parser.add_argument("--conferences", "-c", nargs="+", default=settings.NEED_CONFERENCES_OR_JOURNALS,
                             help="List of conferences to download papers from")
    download_parser.add_argument("--years", "-y", nargs="+", default=settings.NEED_YEAR,
                             help="List of years to download papers for")
    download_parser.add_argument("--output-dir", "-o", default=settings.PDF_SAVE_PATH,
                             help="Output directory for downloaded papers")
    
    # Search papers command
    search_parser = subparsers.add_parser("search", help="Search for papers")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--top-k", "-k", type=int, default=5,
                             help="Number of results to return")
    search_parser.add_argument("--method", "-m", choices=["rag", "db"], default="rag",
                             help="Search method (rag for semantic search, db for database search)")
    search_parser.add_argument("--output", "-o", default=None,
                             help="Output file for search results (JSON format)")
    
    # Process papers command
    process_parser = subparsers.add_parser("process", help="Process papers and generate reports")
    process_parser.add_argument("pdf_paths", nargs="+", help="Paths to PDF files")
    process_parser.add_argument("--api-key", default=None, help="API key for LLM service")
    process_parser.add_argument("--base-url", default=None, help="Base URL for LLM service")
    process_parser.add_argument("--model", default=None, help="Model name for LLM service")
    process_parser.add_argument("--format", "-f", choices=["pdf", "md"], default="pdf",
                             help="Output format for reports")
    
    # Initialize RAG index command
    init_rag_parser = subparsers.add_parser("init-rag", help="Initialize or rebuild the RAG index")
    init_rag_parser.add_argument("--force-rebuild", "-f", action="store_true",
                             help="Force rebuild the index even if it exists")
    
    # List conferences command
    list_conferences_parser = subparsers.add_parser("list-conferences", help="List all available conferences")
    
    # List years command
    list_years_parser = subparsers.add_parser("list-years", help="List all available years for a conference")
    list_years_parser.add_argument("--conference", "-c", default=None, help="Conference name")
    
    args = parser.parse_args()
    
    # Ensure directories exist
    ensure_directories()
    
    # Execute command
    if args.command == "fetch":
        success = fetch_papers(args.conferences, args.years)
        if success:
            print("Papers fetched successfully")
        else:
            print("Error fetching papers")
            
    elif args.command == "download":
        stats = download_papers(args.conferences, args.years, args.output_dir)
        if stats:
            print("Download statistics:")
            for conference, stat in stats.items():
                print(f"  {conference}: {stat['success']}/{stat['total']} ({stat['success_rate']}% success rate)")
        else:
            print("Error downloading papers")
            
    elif args.command == "search":
        results = search_papers(args.query, args.top_k, args.method)
        if results:
            print(f"Found {len(results)} matching papers:")
            for i, paper in enumerate(results):
                print(f"\n{i+1}. {paper.get('title', 'Untitled')}")
                print(f"   Year: {paper.get('year', 'Unknown')}")
                
                if 'distance' in paper:
                    print(f"   Score: {1.0 - paper.get('distance', 0):.2f}")
                    
                if 'authors' in paper and paper['authors']:
                    if isinstance(paper['authors'], list) and isinstance(paper['authors'][0], dict):
                        authors = ", ".join([a.get('name', '') for a in paper['authors']])
                    else:
                        authors = str(paper['authors'])
                    print(f"   Authors: {authors}")
                    
                if 'abstract' in paper or 'abstracts' in paper:
                    abstract = paper.get('abstract', paper.get('abstracts', ''))
                    if len(abstract) > 200:
                        abstract = abstract[:200] + "..."
                    print(f"   Abstract: {abstract}")
                    
            # Save to file if requested
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\nSearch results saved to {args.output}")
        else:
            print("No matching papers found")
            
    elif args.command == "process":
        report_paths = process_papers(
            args.pdf_paths, 
            args.api_key, 
            args.base_url, 
            args.model, 
            args.format
        )
        
        if report_paths:
            print(f"Generated {len(report_paths)} reports:")
            for path in report_paths:
                print(f"  {path}")
        else:
            print("Error processing papers")
            
    elif args.command == "init-rag":
        success = init_rag_index(args.force_rebuild)
        if success:
            print("RAG index initialized successfully")
        else:
            print("Error initializing RAG index")
            
    elif args.command == "list-conferences":
        conferences = list_conferences()
        print("Available conferences:")
        for conf in conferences:
            print(f"  {conf}")
            
    elif args.command == "list-years":
        years = list_years(args.conference)
        if years:
            print("Available years:")
            for conf, conf_years in years.items():
                print(f"  {conf}: {', '.join(conf_years)}")
        else:
            print(f"No years found for conference: {args.conference}")
            
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
