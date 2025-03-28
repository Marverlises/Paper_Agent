# -*- coding: utf-8 -*-
# @Time       : 2025/3/17 15:49
# @Author     : Marverlises
# @File       : utils_md2pdf.py
# @Description: PyCharm
# -*- coding: utf-8 -*-
# @Time       : 2025/3/17 15:01
# @Author     : Marverlises
# @File       : md2pdf.py
# @Description: PyCharm
# !/usr/bin/env python

import os
import argparse
import subprocess
import logging
import shutil
import sys
import tempfile
import platform
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('md2pdf')


def check_pandoc():
    """检查 Pandoc 是否可用"""
    return shutil.which('pandoc') is not None


def check_latex(engine='xelatex'):
    """检查 LaTeX 引擎是否可用"""
    return shutil.which(engine) is not None


def get_default_cjk_font():
    """根据操作系统获取默认中文字体"""
    system = platform.system()
    if system == 'Windows':
        return 'SimSun'
    elif system == 'Darwin':  # macOS
        return 'PingFang SC'
    else:  # Linux 和其他系统
        # 尝试一些常见的中文字体
        fonts = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'AR PL KaitiM GB', 'SimSun']
        for font in fonts:
            # 简单检查字体文件是否存在（不同系统路径不同，这只是一个简单的尝试）
            font_paths = [
                f'/usr/share/fonts/*/"{font}"*',
                f'/usr/local/share/fonts/*/"{font}"*',
                f'~/.fonts/"{font}"*'
            ]
            for path in font_paths:
                try:
                    if subprocess.run(f'ls {path}', shell=True, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE).returncode == 0:
                        return font
                except Exception:
                    pass
        return 'AR PL KaitiM GB'  # 默认回退选项


def preprocess_markdown(input_file, debug=False):
    """预处理 Markdown 文件，处理一些 Pandoc 转换问题"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 处理内联代码，用 \texttt{} 替换 `code`
        def replace_inline_code(match):
            code = match.group(1)
            # 转义特殊字符
            code = code.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
            return f'\\texttt{{{code}}}'

        content = re.sub(r'`([^`]+)`', replace_inline_code, content)

        # 创建临时文件保存处理后的 Markdown
        temp_dir = tempfile.mkdtemp()
        processed_file = os.path.join(temp_dir, os.path.basename(input_file))

        with open(processed_file, 'w', encoding='utf-8') as f:
            f.write(content)

        if debug:
            logger.debug(f"预处理 Markdown 文件: {input_file} -> {processed_file}")

        return processed_file, temp_dir
    except Exception as e:
        logger.error(f"预处理 Markdown 文件失败: {str(e)}")
        return input_file, None  # 如果失败，返回原始文件


def convert_md_to_pdf(input_file, output_file=None, mainfont=None, engine='xelatex', toc=False,
                      verbose=False, debug=False):
    """
    将 Markdown 文件转换为 PDF
    Args:
        input_file (str): 输入的 Markdown 文件路径
        output_file (str, optional): 输出的 PDF 文件路径，默认为输入文件名+.pdf
        mainfont (str): 主字体，默认根据系统自动选择
        engine (str): PDF 引擎，默认为 xelatex
        toc (bool): 是否生成目录
        verbose (bool): 是否显示详细日志
        debug (bool): 是否启用调试模式
    Returns:
        str: 生成的 PDF 文件路径，或 None（失败时）
    """
    if verbose or debug:
        logger.setLevel(logging.DEBUG)

    # 设置默认字体（如果未指定）
    if mainfont is None:
        mainfont = get_default_cjk_font()
        logger.debug(f"使用默认字体: {mainfont}")

    # 检查 Pandoc 是否可用
    if not check_pandoc():
        logger.error("Pandoc 未安装，请先安装 Pandoc")
        logger.info("安装指南: https://pandoc.org/installing.html")
        return None

    # 检查 LaTeX 引擎是否可用
    if not check_latex(engine):
        logger.error(f"{engine} 未安装，请先安装 LaTeX 发行版（如 TeX Live、MiKTeX 或 MacTeX）")
        if platform.system() == 'Linux':
            logger.info("在 Ubuntu/Debian 上可以运行: sudo apt install texlive-xetex texlive-fonts-recommended")
        elif platform.system() == 'Darwin':  # macOS
            logger.info("在 macOS 上可以安装 MacTeX: https://tug.org/mactex/")
        else:  # Windows
            logger.info("在 Windows 上可以安装 MiKTeX: https://miktex.org/download")
        return None

    # 验证输入文件
    if not os.path.exists(input_file):
        logger.error(f"输入文件 {input_file} 不存在")
        return None

    # 设置输出文件
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.pdf'

    # 预处理 Markdown 文件
    processed_file, temp_dir = preprocess_markdown(input_file, debug)

    # 获取输入文件所在目录，用于图片路径
    input_dir = os.path.dirname(os.path.abspath(processed_file))

    # 根据平台确定路径分隔符
    path_separator = ';' if platform.system() == 'Windows' else ':'
    resource_path = f"{input_dir}{path_separator}."

    # 创建临时目录用于存储模板
    template_temp_dir = tempfile.mkdtemp()
    template_path = os.path.join(template_temp_dir, 'template.tex')

    # 改进的 LaTeX 模板，支持中文、数学公式和图片
    template = rf"""
\documentclass[12pt, a4paper]{{article}}
\usepackage{{fontspec}}
\usepackage{{xeCJK}}
\usepackage{{geometry}}
\usepackage{{hyperref}}
\usepackage{{listings}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}  % 支持 \mathbb 等符号
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{grffile}}  % 支持特殊字符文件名
\usepackage{{fancyhdr}} % 页眉页脚
\usepackage{{color}}    % 颜色支持
\usepackage{{xcolor}}   % 增强的颜色支持
\usepackage{{booktabs}} % 更好的表格支持
\usepackage[normalem]{{ulem}}  % 支持删除线 (\sout 命令)
\usepackage{{enumerate}} % 增强的枚举支持
\usepackage{{longtable}} % 跨页表格支持
\usepackage{{array}}     % 增强的表格支持
\usepackage{{multirow}}  % 表格中的多行单元格
\usepackage{{makecell}}  % 表格单元格格式化
\usepackage{{footnote}}  % 脚注支持
\usepackage{{caption}}   % 图表标题支持
\usepackage{{subcaption}} % 子图表支持
\usepackage{{tikz}}      % 绘图支持
\usepackage{{enumitem}}  % 自定义列表样式
\usepackage{{url}}       % URL 支持
\usepackage{{soul}}      % 文本装饰（下划线、高亮等）
\usepackage{{fvextra}}   % 扩展代码块支持
\usepackage{{upquote}}   % 正确渲染引号
\usepackage{{microtype}} % 改进的文本排版
\usepackage{{needspace}} % 避免页面分隔问题
\usepackage{{fancyvrb}}  % 增强的代码块支持

% 设置图片路径
\graphicspath{{{{{input_dir}/}}}}

% 设置中文字体
\setCJKmainfont{{{mainfont}}}
\setCJKsansfont{{{mainfont}}}
\setCJKmonofont{{{mainfont}}}
\CJKspace

% 页面边距
\geometry{{margin=2.5cm}}

% 代码高亮设置
\lstset{{
    basicstyle=\ttfamily\small,
    breaklines=true,
    frame=single,
    numbers=left,
    numberstyle=\tiny\color{{gray}},
    keywordstyle=\color{{blue}},
    commentstyle=\color{{green!50!black}},
    stringstyle=\color{{red}}
}}

% 设置超链接格式
\hypersetup{{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,
    urlcolor=cyan
}}

% 页眉页脚设置
\pagestyle{{fancy}}
\fancyhf{{}}
\lhead{{\leftmark}}
\rhead{{\thepage}}
\cfoot{{\thepage}}

% 定义 \tightlist 命令，兼容 Pandoc 生成的列表
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}

% 确保图片不超出页面边界
\DeclareGraphicsExtensions{{.pdf,.png,.jpg,.jpeg}}
\makeatletter
\def\maxwidth{{\ifdim\Gin@nat@width>\linewidth\linewidth\else\Gin@nat@width\fi}}
\def\maxheight{{\ifdim\Gin@nat@height>\textheight\textheight\else\Gin@nat@height\fi}}
\makeatother
\setkeys{{Gin}}{{width=\maxwidth,height=\maxheight,keepaspectratio}}

\begin{{document}}
$if(title)$
\title{{{os.path.basename(input_file)}}}
\maketitle
$endif$
$if(toc)$
\tableofcontents
\newpage
$endif$
$body$
\end{{document}}
"""
    try:
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template)
    except Exception as e:
        logger.error(f"创建模板文件失败: {str(e)}")
        return None

    # 构建 Pandoc 命令
    cmd = [
        'pandoc',
        processed_file,
        '-o', output_file,
        '--pdf-engine', engine,
        '--template', template_path,
        '-V', f'CJKmainfont={mainfont}',
        '-V', 'monofont=DejaVu Sans Mono',  # 设置等宽字体
        '--standalone',
        '--resource-path', resource_path  # 使用平台适配的资源路径
    ]

    if toc:
        cmd.append('--toc')

    # 添加图像处理相关选项
    cmd.extend([
        '--wrap=none',  # 避免行换行影响图片渲染
        '--extract-media=.',  # 提取嵌入式媒体文件
    ])

    if verbose or debug:
        cmd.append('--verbose')  # 生成详细日志

    # 显示实际要执行的命令 (用于调试)
    cmd_str = ' '.join(cmd)
    logger.info(f"执行命令: {cmd_str}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        if verbose or debug:
            logger.debug(f"Pandoc 输出: {result.stdout}")
            if result.stderr:
                logger.debug(f"Pandoc 错误: {result.stderr}")
        logger.info(f"PDF 成功生成: {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Pandoc 转换失败: {str(e)}")
        if e.stderr:
            logger.error(f"错误详情: {e.stderr}")

            # 尝试提取更具体的LaTeX错误信息
            if "Error producing PDF" in e.stderr:
                latex_errors = []
                for line in e.stderr.split('\n'):
                    if line.startswith('!') or "Undefined control sequence" in line:
                        latex_errors.append(line.strip())

                if latex_errors:
                    logger.error("LaTeX 错误:")
                    for err in latex_errors:
                        logger.error(f"  {err}")

                    # 提供可能的解决方案
                    if "Undefined control sequence" in e.stderr:
                        logger.info("可能的解决方案: 某些LaTeX命令未定义，请检查是否需要额外的LaTeX包")

        return None
    except Exception as e:
        logger.error(f"转换过程中发生错误: {str(e)}")
        return None
    finally:
        # 清理临时文件
        try:
            if template_temp_dir:
                shutil.rmtree(template_temp_dir)
            if temp_dir:
                shutil.rmtree(temp_dir)
        except Exception as e:
            logger.warning(f"清理临时文件失败: {str(e)}")


def markdown_to_pdf(input_file, output_file=None, mainfont=None, engine='xelatex', toc=False, open_pdf=False):
    """
    简单的工具函数，将Markdown文件转换为PDF

    Args:
        input_file (str): 输入的Markdown文件路径
        output_file (str, optional): 输出的PDF文件路径，默认为输入文件名.pdf
        mainfont (str, optional): 主字体，默认自动选择
        engine (str, optional): PDF引擎，默认为xelatex
        toc (bool, optional): 是否生成目录，默认为False
        open_pdf (bool, optional): 转换完成后是否自动打开PDF，默认为False

    Returns:
        str: 生成的PDF文件路径，失败时返回None
    """
    # 确保输入文件是绝对路径
    input_file = os.path.abspath(input_file)

    # 如果没有指定输出文件，使用默认路径
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + '.pdf'
    else:
        # 如果提供了相对输出路径，确保它是绝对路径
        if not os.path.isabs(output_file):
            output_file = os.path.abspath(output_file)

    # 调用核心转换函数
    result = convert_md_to_pdf(
        input_file=input_file,
        output_file=output_file,
        mainfont=mainfont,
        engine=engine,
        toc=toc
    )

    # 如果转换成功且需要打开文件
    if result and open_pdf:
        try:
            if platform.system() == 'Windows':
                os.startfile(result)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.run(['open', result], check=False)
            else:  # Linux
                subprocess.run(['xdg-open', result], check=False)
        except Exception as e:
            logger.debug(f"无法自动打开PDF文件: {e}")

    return result


# 保留命令行接口，以便脚本仍可作为命令行工具使用
def main():
    parser = argparse.ArgumentParser(description='将 Markdown 转换为 PDF')
    parser.add_argument('input', help='输入 Markdown 文件路径', nargs='?')
    parser.add_argument('-o', '--output', help='输出 PDF 文件路径（默认：输入文件名+.pdf）')
    parser.add_argument('--mainfont', help='指定主字体（默认：根据系统自动选择）')
    parser.add_argument('--engine', default='xelatex', choices=['xelatex', 'pdflatex', 'lualatex'],
                        help='PDF 引擎（默认：xelatex）')
    parser.add_argument('--toc', action='store_true', help='生成目录')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细日志')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose or args.debug:
        logger.setLevel(logging.DEBUG)

    # 使用命令行参数或提示用户输入
    input_file = args.input
    if not input_file:
        # 尝试使用示例文件
        default_examples = [
            "./example.md",
            "./README.md",
            "./docs/example.md"
        ]

        for example in default_examples:
            if os.path.exists(example):
                input_file = example
                logger.info(f"使用示例文件: {input_file}")
                break

        # 如果没有找到示例文件，提示用户输入
        if not input_file:
            logger.info("未指定输入文件，请提供 Markdown 文件路径:")
            try:
                input_file = input("> ")
            except (KeyboardInterrupt, EOFError):
                logger.info("\n已取消操作")
                sys.exit(0)

    # 调用简化的工具函数
    result = markdown_to_pdf(
        input_file=input_file,
        output_file=args.output,
        mainfont=args.mainfont,
        engine=args.engine,
        toc=args.toc,
        open_pdf=True
    )

    if not result:
        logger.error("转换失败")
        sys.exit(1)
    else:
        logger.info(f"转换成功: {result}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n操作已中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行过程中发生错误: {str(e)}")
        sys.exit(1)