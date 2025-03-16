# -*- coding: utf-8 -*-
# @Time       : 2025/3/6 8:50
# @Author     : Marverlises
# @File       : paper_processor.py
# @Description: Paper processor for analyzing and generating reports
import os
import logging
import json
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import requests
from md2pdf.core import md2pdf

from modules.llm_infer import LLMInfer
from modules.utils import PDFUtils
from config import settings

logger = logging.getLogger(__name__)

class PaperProcessor:
    """
    Paper processor for analyzing papers and generating reports.
    Handles PDF segmentation, text extraction, and report generation.
    """
    # Service URL for PDF processing, configurable via settings
    SERVICE_URL = getattr(settings, "PDF_SERVICE_URL", "http://localhost:5060")
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model_name: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the paper processor.
        
        Args:
            api_key (str, optional): API key for the LLM service. Defaults to None.
            base_url (str, optional): Base URL for the LLM service. Defaults to None.
            model_name (str, optional): Name of the model to use. Defaults to None.
            output_dir (str, optional): Directory to save output reports. Defaults to None.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.output_dir = output_dir or os.path.join(settings.ANALYSIS_REPORT_PATH, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize cache to avoid reprocessing
        self.segmentation_cache = {}
        
    @classmethod
    def segment_pdf(cls, 
                    file_paths: List[str], 
                    save_path: Optional[str] = None) -> Tuple[bool, List[str]]:
        """
        Segment PDF files into structured content blocks.
        
        Args:
            file_paths (List[str]): List of PDF file paths to segment
            save_path (str, optional): Directory to save segmentation results. Defaults to None.
            
        Returns:
            Tuple[bool, List[str]]: Success status and list of failed file paths
        """
        if not save_path:
            save_path = tempfile.mkdtemp(prefix="paper_segmentation_")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        failed_paths = []
        for file_path in file_paths:
            try:
                with open(file_path, "rb") as stream:
                    files = {"file": stream}
                    
                    # Send file to segmentation service
                    logger.info(f"Sending {file_path} to segmentation service at {cls.SERVICE_URL}")
                    results = requests.post(f"{cls.SERVICE_URL}", files=files)
                    
                    if results.status_code != 200:
                        logger.error(f"Failed to segment PDF: {file_path}. Status code: {results.status_code}")
                        failed_paths.append(file_path)
                        continue
                        
                    results_list = results.json()

                # Save results to JSON
                base_filename = os.path.basename(file_path).split('.')[0]
                save_json_path = os.path.join(save_path, f"{base_filename}.json")
                with open(save_json_path, "w", encoding="utf-8") as f:
                    json.dump(results_list, f, ensure_ascii=False, indent=2)
                logger.info(f"Successfully saved segmentation to {save_json_path}")
            except Exception as e:
                failed_paths.append(file_path)
                logger.error(f"Error segmenting PDF {file_path}: {e}")
                
        success = len(failed_paths) == 0
        return success, failed_paths
        
    def extract_figures_and_tables(self, 
                                  pdf_path: str, 
                                  segment_json_path: str,
                                  output_dir: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Extract figures and tables from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            segment_json_path (str): Path to the segmentation JSON file
            output_dir (str, optional): Directory to save extracted figures and tables. Defaults to None.
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping figure/table references to file paths
        """
        if not output_dir:
            paper_name = os.path.basename(pdf_path).split('.')[0]
            output_dir = os.path.join(self.output_dir, paper_name)
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Load segmentation data
        with open(segment_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Find tables and figures with captions
        figures_and_tables = []
        for index, item in enumerate(data):
            if item['type'] in ['Table', 'Picture']:
                # Check for captions before or after
                if index + 1 < len(data) and data[index + 1]['type'] == 'Caption':
                    figures_and_tables.append({'data': item, 'caption': data[index + 1]})
                elif index - 1 >= 0 and data[index - 1]['type'] == 'Caption':
                    figures_and_tables.append({'data': item, 'caption': data[index - 1]})
                else:
                    # If no caption is found, check if text contains figure/table reference
                    if ('图' in item['text'] or '表' in item['text'] or 
                        'figure' in item['text'].lower() or 'table' in item['text'].lower() or 
                        'fig' in item['text'].lower() or 'tab' in item['text'].lower()):
                        figures_and_tables.append({'data': item, 'caption': item})
                    else:
                        logger.warning(f"Item without caption detected: {item['text']}")
        
        # Extract and save figures and tables
        picture_index = 1
        table_index = 1
        extracted_items = {'Picture': [], 'Table': []}
        
        for item in figures_and_tables:
            data_item = item['data']
            caption_item = item['caption']
            
            # Create bounding box for extraction
            boxes = [
                [data_item['left'], data_item['top'], 
                 data_item['left'] + data_item['width'], data_item['top'] + data_item['height'] + 2],
                [caption_item['left'], caption_item['top'], 
                 caption_item['left'] + caption_item['width'], caption_item['top'] + caption_item['height'] + 2]
            ]
            
            combined_box = PDFUtils.combine_boxes(boxes)
            page_num = data_item['page_number'] - 1
            
            # Get PDF page
            page = PDFUtils.get_pdf_page(pdf_path, page_num)
            
            # Create file path based on item type
            if data_item['type'] == 'Picture':
                item_path = os.path.join(output_dir, f"Picture_{picture_index}.png")
                reference = f"Figure-{picture_index}"
                picture_index += 1
            else:  # Table
                item_path = os.path.join(output_dir, f"Table_{table_index}.png")
                reference = f"Table-{table_index}"
                table_index += 1
                
            # Clip and save the item
            PDFUtils.clip_object(combined_box, page, item_path)
            extracted_items[data_item['type']].append({
                'reference': reference,
                'path': item_path,
                'caption': caption_item['text'] if 'text' in caption_item else ""
            })
            
        return extracted_items
        
    def analyze_paper(self, 
                     segment_json_path: str,
                     extracted_items: Optional[Dict[str, List[Dict[str, str]]]] = None,
                     system_prompt: str = "",
                     max_retries: int = 3) -> str:
        """
        Analyze a paper using LLM.
        
        Args:
            segment_json_path (str): Path to the segmentation JSON file
            extracted_items (Dict[str, List[Dict[str, str]]], optional): Extracted figures and tables. Defaults to None.
            system_prompt (str, optional): System prompt for the LLM. Defaults to "".
            max_retries (int, optional): Maximum number of retries for LLM inference. Defaults to 3.
            
        Returns:
            str: Generated analysis
        """
        if not self.api_key or not self.base_url or not self.model_name:
            raise ValueError("API key, base URL, and model name must be provided")
            
        # Load segmented paper data
        with open(segment_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Convert data to text format
        text_data = [f"{item['type']}: {item['text']}" for item in data]
        
        # Define a more detailed prompt for the analysis
        analysis_prompt = (
            "请非常详细地分析这篇论文，包括以下几个方面：\n"
            "1. 研究背景与问题：论文研究的问题是什么？为什么这个问题很重要？\n"
            "2. 主要创新点：论文的主要贡献和创新点是什么？\n"
            "3. 方法学：论文提出了什么新方法或技术？如何解决研究问题？\n"
            "4. 实验结果：论文通过什么实验验证了方法的有效性？结果如何？\n"
            "5. 总结与展望：论文的主要结论是什么？有什么局限性或未来工作？\n\n"
            "在回答时，请在适当的位置插入论文中的图表，格式为单独的一行：>>> Figure-X <<<或>>> Table-Y <<<\n"
            "请用中文回答，数学公式请用$或$$包裹。\n\n"
            f"论文内容：\n{text_data}"
        )
        
        # Try to get response with retries
        for attempt in range(max_retries):
            try:
                response_info = LLMInfer.API_infer(
                    self.api_key, 
                    self.base_url, 
                    [analysis_prompt], 
                    self.model_name, 
                    system_prompt or "You are a helpful academic assistant specialized in analyzing research papers."
                )
                
                if response_info and len(response_info) > 0:
                    return response_info[0]['generated_text']
                else:
                    logger.warning(f"Empty response on attempt {attempt+1}/{max_retries}")
            except Exception as e:
                logger.error(f"Error in LLM inference (attempt {attempt+1}/{max_retries}): {e}")
                
        # If all retries failed
        return "无法分析论文。请检查API配置或重试。"

    def generate_report(self, 
                       analysis_text: str,
                       extracted_items: Dict[str, List[Dict[str, str]]],
                       paper_metadata: Dict[str, Any],
                       output_format: str = 'pdf') -> str:
        """
        Generate a report from paper analysis.
        
        Args:
            analysis_text (str): Analysis text generated by LLM
            extracted_items (Dict[str, List[Dict[str, str]]]): Extracted figures and tables
            paper_metadata (Dict[str, Any]): Paper metadata (title, authors, etc.)
            output_format (str, optional): Output format ('pdf' or 'md'). Defaults to 'pdf'.
            
        Returns:
            str: Path to the generated report
        """
        # Create report title from paper metadata
        title = paper_metadata.get('title', 'Untitled Paper')
        authors = paper_metadata.get('authors', [])
        if isinstance(authors, list) and authors and isinstance(authors[0], dict):
            author_names = [author.get('name', '') for author in authors]
            author_text = ', '.join(author_names)
        else:
            author_text = str(authors)
            
        # Create report content with metadata
        report_content = f"# {title}\n\n"
        report_content += f"**Authors:** {author_text}\n\n"
        
        if 'year' in paper_metadata:
            report_content += f"**Year:** {paper_metadata['year']}\n\n"
            
        if 'keywords' in paper_metadata and paper_metadata['keywords']:
            keywords = paper_metadata['keywords']
            if isinstance(keywords, list):
                keywords_text = ', '.join(keywords)
            else:
                keywords_text = str(keywords)
            report_content += f"**Keywords:** {keywords_text}\n\n"
            
        report_content += "---\n\n"
        report_content += f"## 论文分析\n\n"
        
        # Process analysis text to replace figure/table references
        lines = analysis_text.split('\n')
        for i, line in enumerate(lines):
            if '>>>' in line and '<<<' in line:
                ref_match = None
                
                # Check for figure references
                if 'Figure-' in line:
                    figure_num = line.split('Figure-')[1].split()[0].strip('<<<').strip()
                    try:
                        figure_idx = int(figure_num) - 1
                        if figure_idx < len(extracted_items.get('Picture', [])):
                            figure_path = extracted_items['Picture'][figure_idx]['path']
                            # Convert to relative path for markdown
                            rel_path = os.path.relpath(figure_path, self.output_dir)
                            lines[i] = f"![Figure {figure_num}]({rel_path})"
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error processing figure reference: {e}")
                        
                # Check for table references
                elif 'Table-' in line:
                    table_num = line.split('Table-')[1].split()[0].strip('<<<').strip()
                    try:
                        table_idx = int(table_num) - 1
                        if table_idx < len(extracted_items.get('Table', [])):
                            table_path = extracted_items['Table'][table_idx]['path']
                            # Convert to relative path for markdown
                            rel_path = os.path.relpath(table_path, self.output_dir)
                            lines[i] = f"![Table {table_num}]({rel_path})"
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Error processing table reference: {e}")
        
        # Join processed lines back into text
        report_content += '\n'.join(lines)
        
        # Save as markdown
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join([c if c.isalnum() else "_" for c in title])[:50]  # Create safe filename
        
        md_filename = f"report_{safe_title}_{timestamp}.md"
        md_path = os.path.join(self.output_dir, md_filename)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        logger.info(f"Saved markdown report to {md_path}")
        
        # Convert to PDF if requested
        if output_format.lower() == 'pdf':
            pdf_filename = md_filename.replace('.md', '.pdf')
            pdf_path = os.path.join(self.output_dir, pdf_filename)
            
            try:
                md2pdf(pdf_path, md_file_path=md_path, base_url=self.output_dir)
                logger.info(f"Generated PDF report: {pdf_path}")
                return pdf_path
            except Exception as e:
                logger.error(f"Error generating PDF: {e}")
                return md_path
        
        return md_path
        
    def process_paper(self, 
                     pdf_path: str, 
                     paper_metadata: Optional[Dict[str, Any]] = None,
                     output_format: str = 'pdf') -> str:
        """
        Process a single paper from PDF to final report.
        
        Args:
            pdf_path (str): Path to the PDF file
            paper_metadata (Dict[str, Any], optional): Paper metadata. Defaults to None.
            output_format (str, optional): Output format ('pdf' or 'md'). Defaults to 'pdf'.
            
        Returns:
            str: Path to the generated report
        """
        logger.info(f"Processing paper: {pdf_path}")
        paper_name = os.path.basename(pdf_path).split('.')[0]
        
        # Create processing directory
        paper_dir = os.path.join(self.output_dir, paper_name)
        os.makedirs(paper_dir, exist_ok=True)
        
        # Step 1: Segment the PDF
        success, _ = self.segment_pdf([pdf_path], save_path=paper_dir)
        if not success:
            logger.error(f"Failed to segment PDF: {pdf_path}")
            return ""
            
        segment_json_path = os.path.join(paper_dir, f"{paper_name}.json")
        
        # Step 2: Extract figures and tables
        extracted_items = self.extract_figures_and_tables(pdf_path, segment_json_path, paper_dir)
        
        # Step 3: Analyze the paper
        analysis_text = self.analyze_paper(segment_json_path, extracted_items)
        
        # Step 4: Generate the report
        if not paper_metadata:
            paper_metadata = {"title": paper_name}
            
        report_path = self.generate_report(analysis_text, extracted_items, paper_metadata, output_format)
        
        return report_path
        
    def batch_process_papers(self, 
                            pdf_paths: List[str], 
                            metadata_list: Optional[List[Dict[str, Any]]] = None,
                            output_format: str = 'pdf') -> List[str]:
        """
        Process multiple papers in batch.
        
        Args:
            pdf_paths (List[str]): List of PDF file paths
            metadata_list (List[Dict[str, Any]], optional): List of paper metadata. Defaults to None.
            output_format (str, optional): Output format ('pdf' or 'md'). Defaults to 'pdf'.
            
        Returns:
            List[str]: List of paths to generated reports
        """
        if not metadata_list:
            metadata_list = [None] * len(pdf_paths)
            
        if len(pdf_paths) != len(metadata_list):
            raise ValueError("Number of PDF paths and metadata entries must match")
            
        report_paths = []
        for i, (pdf_path, metadata) in enumerate(zip(pdf_paths, metadata_list)):
            logger.info(f"Processing paper {i+1}/{len(pdf_paths)}: {pdf_path}")
            try:
                report_path = self.process_paper(pdf_path, metadata, output_format)
                if report_path:
                    report_paths.append(report_path)
            except Exception as e:
                logger.error(f"Error processing paper {pdf_path}: {e}")
                
        return report_paths
