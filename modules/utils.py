# -*- coding: utf-8 -*-
# @Time       : 2025/2/9 11:44
# @Author     : Marverlises
# @File       : utils.py
# @Description: PyCharm
import logging
import random
import time
import requests
import json
import os
import io
import pdfplumber
from lxml import html


class Utils:
    @staticmethod
    def clean_text(text):
        """ Helper function to clean and concatenate text from HTML elements. """
        return " ".join(text.replace('\n', ' ').split())

    @staticmethod
    def convert_data(value):
        """自动将数据转换为适合插入数据库的格式"""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        elif isinstance(value, (int, float, str)):
            return value
        else:
            return str(value)

    @staticmethod
    def download_pdf_from_url(pdf_url, save_path, save_file_name):
        try:
            response = requests.get(pdf_url)
            if response.status_code == 200:
                send_headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
                    "Connection": "keep-alive",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.8"}
                response = requests.get(pdf_url, headers=send_headers, timeout=30)
                bytes_io = io.BytesIO(response.content)
                with open(os.path.join(save_path, f"{save_file_name}.pdf"), mode='wb') as f:
                    f.write(bytes_io.getvalue())
                logging.info(f"PDF successfully downloaded from {pdf_url} and saved to {save_path}.")
                return True
            else:
                logging.error(f"Failed to download PDF from {pdf_url}. Status code: {response.status_code}")
                return False
        except Exception as e:
            logging.error(f"Error occurred while downloading PDF from {pdf_url}: {str(e)}")
            return False


class RequestUtils:
    headers_list = [
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36"},
        {
            "User-Agent": "Mozilla/5.0 (iPad; CPU OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1"},
        {
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 12_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148"},
        {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:54.0) Gecko/20100101 Firefox/54.0"},
        {"User-Agent": "Mozilla/5.0 (Android 8.0.0; Mobile; rv:61.0) Gecko/61.0 Firefox/61.0"},
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko"},
        {
            "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.76 Mobile Safari/537.36"},
        {"User-Agent": "Mozilla/5.0 (X11; Linux i586; rv:31.0) Gecko/20100101 Firefox/31.0"},
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36"},
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.112 Safari/537.36"},
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/7046A194A"},
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_6_8) AppleWebKit/534.58.2 (KHTML, like Gecko) Version/5.1.8 Safari/534.58.2"},
        {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:60.0) Gecko/20100101 Firefox/60.0"},
        {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:55.0) Gecko/20100101 Firefox/55.0"},
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36"}
    ]

    @staticmethod
    def make_request(url, max_attempts=3):
        """
        发送HTTP请求
        :param url:             请求的URL
        :param max_attempts:    最大尝试次数
        :return:                HTTP响应
        """
        # 随机休息1-3秒
        time.sleep(random.randint(1, 3))
        for attempt in range(max_attempts):
            headers = random.choice(RequestUtils.headers_list)
            proxy = RequestUtils.get_random_proxy()
            try:
                response = requests.get(url, headers=headers, proxies=proxy, timeout=3)
                response.raise_for_status()
                return response
            except (requests.RequestException, requests.Timeout) as e:
                logging.info(f"请求失败，正在重试...（尝试次数：{attempt + 1}）")
                if attempt == max_attempts - 1:
                    raise Exception(f"请求失败已达最大重试次数：{e}")
        return None

    @staticmethod
    def get_random_proxy():
        """
        获取随机一个代理
        :return:
        """

        def fetch_proxy_list():
            """
            获取代理IP列表
            :return:
            """
            # TODO
            return []

        proxy_list = fetch_proxy_list()
        if not proxy_list:
            return None
        proxy = random.choice(proxy_list)
        proxy = {
            'http': f'http://{proxy}'
        }
        logging.info(f"Using proxy: {proxy}")
        return proxy


class ArxivUtils:
    @staticmethod
    def fetch_abstract(link):
        """
        从arXiv链接中获取摘要。
        :param link:    str, arXiv链接
        :return:        str, 摘要内容
        """
        if not link:
            return 'no abstract'
        # 访问链接，获取摘要内容
        # 使用该代理尝试访问链接3次，失败则更换代理,如果三次都失败，则跳过
        response = RequestUtils.make_request(link)
        if not response:
            return 'no abstract'

        # 解析HTML
        tree = html.fromstring(response.content)
        abstract_xpath = "//*[@id='abs']/blockquote"
        abstract_text = tree.xpath(abstract_xpath + "/text()")
        abstract = Utils.clean_text(" ".join(abstract_text))

        return abstract

    @staticmethod
    def fetch_arxiv_link_and_abstract(title):
        """
        从arXiv搜索标题并返回第一个搜索结果的链接和摘要。
        :param title: str, 论文标题
        :return:
        """
        # 构建搜索URL
        query = "+".join(title.split())
        url = f"https://arxiv.org/search/?query={query}&searchtype=all&abstracts=show&order=-announced_date_first&size=50"

        # 发送HTTP请求
        response = RequestUtils.make_request(url)
        response.raise_for_status()  # 确保请求成功

        # 解析HTML
        tree = html.fromstring(response.content)

        # XPath定位第一个搜索结果的li标签
        first_result_xpath = "//*[@id='main-container']/div[2]/ol/li[1]"
        first_result = tree.xpath(first_result_xpath)

        if not first_result:
            return 'no link', 'no abstract'

        # 获取第一个p标签中的文本，并与原标题进行比较
        title_text_xpath = ".//p[@class='title is-5 mathjax']/descendant-or-self::*/text()"
        title_parts = first_result[0].xpath(title_text_xpath)
        found_title = Utils.clean_text("".join(title_parts))

        if title == found_title:
            # 如果标题匹配，获取第一个p标签下的a标签的href属性
            link_xpath = ".//div/p/a/@href"
            link = first_result[0].xpath(link_xpath)[0]
            logging.info(f"Title: {title}, Link: {link}")
            # 获取摘要
            abstract = ArxivUtils.fetch_abstract(link)
            return link, abstract
        else:
            logging.info(f"Title not found: {title}")
            return 'no link', 'no abstract'


class PDFUtils:
    @staticmethod
    def clip_object(bbox, page, clip_path):
        """
        从PDF页面中裁剪对象
        :param bbox:        list, 裁剪框的坐标
        :param page:        pdfplumber.Page, PDF页面对象
        :param clip_path:   str, 裁剪后的对象保存路径
        :return:
        """
        # 裁剪对象
        x0, y0, x1, y1 = bbox
        clip = page.within_bbox((x0, y0, x1, y1))
        clip.to_image(resolution=600).save(clip_path)

    @staticmethod
    def combine_boxes(boxes):
        """
        合并多个裁剪框
        :param boxes:   list, 裁剪框列表  [[x0, y0, x1, y1], ...]
        :return:        list, 合并后的裁剪框
        """
        x0 = min([box[0] for box in boxes])
        y0 = min([box[1] for box in boxes])
        x1 = max([box[2] for box in boxes])
        y1 = max([box[3] for box in boxes])
        return [x0, y0, x1, y1]

    # 使用Pyplumber读取PDF的某一页
    @staticmethod
    def get_pdf_page(pdf_path, page_number):
        """
        读取 PDF 文件并返回指定页面的 Page 对象。

        参数：
        - pdf_path: PDF 文件的路径（字符串）
        - page_number: 要读取的页面编号（从 0 开始计数）

        返回：
        - pdfplumber 的 Page 对象
        """
        try:
            # 打开 PDF 文件
            with pdfplumber.open(pdf_path) as pdf:
                # 检查页面编号是否有效
                if page_number < 0 or page_number >= len(pdf.pages):
                    raise ValueError(f"页面编号无效。文件共有 {len(pdf.pages)} 页，编号从 0 到 {len(pdf.pages) - 1}。")

                # 返回指定页面对象
                page = pdf.pages[page_number]
                return page

        except FileNotFoundError:
            raise FileNotFoundError("找不到指定的 PDF 文件，请检查路径是否正确。")
        except Exception as e:
            raise Exception(f"发生错误：{str(e)}")


if __name__ == '__main__':
    # links = ["http://arxiv.org/pdf/2312.10707"]
    # for link in links:
    #     Utils.download_pdf_from_url(link, '../data/', 'test')

    clip_test_path = r'/ai/teacher/mwt/code/by/project/Paper_Agent/pdf_analyzer/test.json'
    pdf_path = r'/ai/teacher/mwt/code/by/project/Paper_Agent/pdf_analyzer/test_pdfs/Academic-paper.pdf'
    with open(clip_test_path, 'r') as f:
        data = json.load(f)

    # find all of the boxes in the data with type 'Table' or 'Picture'
    need_data = []
    for index, item in enumerate(data):
        if item['type'] == 'Table' or item['type'] == 'Picture':
            # if the next item is a caption, or the previous item is a caption, construct a dict
            if index + 1 < len(data) and data[index + 1]['type'] == 'Caption':
                need_data.append({'data': item, 'caption': data[index + 1]})
            elif index - 1 >= 0 and data[index - 1]['type'] == 'Caption':
                need_data.append({'data': item, 'caption': data[index - 1]})
            else:
                print(item['text'])
    # combine the boxes
    picture_index = 1
    table_index = 1
    for index, item in enumerate(need_data):
        data = item['data']
        caption = item['caption']
        # combine the boxes
        boxes = [[data['left'], data['top'], data['left'] + data['width'], data['top'] + data['height'] + 2],
                 [caption['left'], caption['top'], caption['left'] + caption['width'],
                  caption['top'] + caption['height'] + 2]]

        combined_box = PDFUtils.combine_boxes(boxes)
        print(combined_box)
        page_num = data['page_number'] - 1

        page = PDFUtils.get_pdf_page(pdf_path, page_num)
        # clip the object
        if data['type'] == 'Picture':
            clip_path = f'../data/{data["type"]}_{picture_index}.png'
            picture_index += 1
        if data['type'] == 'Table':
            clip_path = f'../data/{data["type"]}_{table_index}.png'
            table_index += 1
        PDFUtils.clip_object(combined_box, page, clip_path)


