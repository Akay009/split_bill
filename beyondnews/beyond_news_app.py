import streamlit as st
import trafilatura
import justext
import html2text
import inscriptis
import requests
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="HTML to Text Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for better styling with improved readability
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .library-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #dee2e6;
    }
    .content-box {
        background-color: #ffffff;
        border: 2px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        color: #212529;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.5;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }
    .pdf-link {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #856404;
        font-weight: 500;
    }
    .pdf-link a {
        color: #0066cc;
        text-decoration: underline;
        font-weight: bold;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #155724;
        font-weight: 600;
        font-size: 16px;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #856404;
        font-weight: 600;
        font-size: 16px;
    }
    .error-box {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #721c24;
        font-weight: 600;
        font-size: 16px;
    }
    .info-box {
        background-color: #e7f3ff;
        border: 2px solid #007bff;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #004085;
        font-weight: 500;
    }
    .stats-text {
        color: #6c757d;
        font-style: italic;
        font-size: 14px;
        margin-bottom: 10px;
    }
    .section-title {
        color: #495057;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 10px;
        margin-top: 20px;
    }
    /* Override Streamlit's default text colors */
    .stMarkdown p {
        color: #212529 !important;
    }
    .stText {
        color: #212529 !important;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def find_pdf_links(html, base_url):
    """Find all PDF links in the HTML content"""
    soup = BeautifulSoup(html, 'html.parser')
    pdf_links = []

    for link in soup.find_all('a', href=True):
        href = link['href']
        text = link.get_text(strip=True)
        title = link.get('title', '')

        if href.lower().endswith('.pdf') or '.pdf' in href.lower():
            full_url = urljoin(base_url, href)
            pdf_links.append({
                'url': full_url,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'title': title
            })

    return pdf_links


def extract_basic_metadata(html):
    """Extract basic metadata using BeautifulSoup"""
    soup = BeautifulSoup(html, 'html.parser')

    # Title extraction
    title = "Not found"
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text(strip=True)
    else:
        og_title = soup.find('meta', attrs={'property': 'og:title'})
        if og_title and og_title.get('content'):
            title = og_title.get('content')

    # Date extraction
    date = "Not found"
    date_selectors = [
        {'name': 'date'},
        {'property': 'article:published_time'},
        {'name': 'pubdate'},
        {'property': 'article:modified_time'},
        {'name': 'last-modified'}
    ]

    for selector in date_selectors:
        meta_date = soup.find('meta', attrs=selector)
        if meta_date and meta_date.get('content'):
            date = meta_date.get('content')
            break

    return title, date


def format_text_preview(text, max_length=1000):
    """Format text for preview with length limit"""
    if not text:
        return "No content extracted"

    if len(text) > max_length:
        return text[:max_length] + "\n\n... (content truncated for display)"
    return text


def display_library_results(library_name, title, date, content, pdf_links, has_pdf, additional_info=None):
    """Display results for a library in a formatted way"""

    st.markdown(f'<div class="library-header">{library_name}</div>', unsafe_allow_html=True)

    # Metadata section
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="section-title">📄 Title</div>', unsafe_allow_html=True)
        if title and title != "Not found":
            st.markdown(f'<div class="success-box">✅ {title}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">❌ Title not found</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">📅 Date</div>', unsafe_allow_html=True)
        if date and date != "Not found":
            st.markdown(f'<div class="success-box">✅ {date}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">❌ Date not found</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="section-title">📎 PDF Links</div>', unsafe_allow_html=True)
        if has_pdf:
            st.markdown(f'<div class="success-box">✅ Found {len(pdf_links)} PDF(s)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warning-box">❌ No PDFs found</div>', unsafe_allow_html=True)

    # PDF Links details
    if has_pdf and pdf_links:
        st.markdown('<div class="section-title">🔗 PDF Link Details</div>', unsafe_allow_html=True)
        for i, pdf in enumerate(pdf_links, 1):
            st.markdown(f"""
            <div class="pdf-link">
                <strong>PDF {i}:</strong><br>
                <strong>URL:</strong> <a href="{pdf['url']}" target="_blank">{pdf['url']}</a><br>
                <strong>Text:</strong> {pdf['text']}<br>
                <strong>Title:</strong> {pdf['title'] if pdf['title'] else 'No title'}
            </div>
            """, unsafe_allow_html=True)

    # Content section
    st.markdown('<div class="section-title">📝 Extracted Content</div>', unsafe_allow_html=True)

    # Show content statistics
    if content:
        word_count = len(content.split())
        char_count = len(content)
        st.markdown(
            f'<div class="stats-text">Content Statistics: {char_count:,} characters, {word_count:,} words</div>',
            unsafe_allow_html=True)

    content_preview = format_text_preview(content, 100000)
    st.markdown(f'<div class="content-box">{content_preview}</div>', unsafe_allow_html=True)

    # Additional info if provided
    if additional_info:
        st.markdown('<div class="section-title">ℹ️ Additional Information</div>', unsafe_allow_html=True)
        info_text = "<br>".join([f"<strong>{key}:</strong> {value}" for key, value in additional_info.items()])
        st.markdown(f'<div class="info-box">{info_text}</div>', unsafe_allow_html=True)

    st.markdown("---")


# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🔍 HTML to Text Analyzer</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; font-size: 1.2rem; color: #495057; font-weight: 500;">
    Enter a URL below to analyze and compare how different libraries extract content, metadata, and detect PDF links.
    </div>
    """, unsafe_allow_html=True)

    # URL input
    url = st.text_input(
        "🌐 Enter URL to analyze:",
        placeholder="https://example.com",
        help="Enter any valid URL to analyze its content"
    )

    # Analysis button
    if st.button("🚀 Analyze Website", type="primary", use_container_width=True):
        if not url:
            st.markdown('<div class="error-box">⚠️ Please enter a URL to analyze.</div>', unsafe_allow_html=True)
            return

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        try:
            with st.spinner("🔄 Fetching and analyzing the website..."):
                # Fetch HTML content
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                html_content = response.text

                # Find PDF links
                pdf_links = find_pdf_links(html_content, url)
                has_pdf = len(pdf_links) > 0

                st.markdown(f'<div class="success-box">✅ Successfully fetched content from: {url}</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="info-box"><strong>Page size:</strong> {len(html_content):,} characters</div>',
                            unsafe_allow_html=True)

                # Create tabs for different libraries
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "🎯 Trafilatura",
                    "📰 JusText",
                    "📝 HTML2Text",
                    "📋 Inscriptis",
                    "🔧 BeautifulSoup",
                    "📊 Summary"
                ])

                # Store results for summary
                results = {}

                # 1. Trafilatura
                with tab1:
                    try:
                        trafilatura_text = trafilatura.extract(html_content, include_comments=False,
                                                               include_tables=True,include_links=True,with_metadata=True)
                        trafilatura_metadata = trafilatura.extract_metadata(html_content)

                        title = trafilatura_metadata.title if trafilatura_metadata and trafilatura_metadata.title else "Not found"
                        date = trafilatura_metadata.date if trafilatura_metadata and trafilatura_metadata.date else "Not found"

                        additional_info = {}
                        if trafilatura_metadata:
                            if trafilatura_metadata.author:
                                additional_info["Author"] = trafilatura_metadata.author
                            if trafilatura_metadata.sitename:
                                additional_info["Site Name"] = trafilatura_metadata.sitename
                            if trafilatura_metadata.description:
                                additional_info["Description"] = trafilatura_metadata.description

                        display_library_results("Trafilatura", title, date, trafilatura_text, pdf_links, has_pdf,
                                                additional_info)

                        results['Trafilatura'] = {
                            'title': title,
                            'date': date,
                            'content_length': len(trafilatura_text) if trafilatura_text else 0,
                            'has_metadata': bool(trafilatura_metadata and trafilatura_metadata.title)
                        }

                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Trafilatura error: {str(e)}</div>',
                                    unsafe_allow_html=True)
                        results['Trafilatura'] = {'error': str(e)}

                # 2. JusText
                with tab2:
                    try:
                        justext_paragraphs = justext.justext(html_content, justext.get_stoplist('english'))
                        justext_text = "\n\n".join([p.text for p in justext_paragraphs if not p.is_boilerplate])

                        title, date = extract_basic_metadata(html_content)

                        additional_info = {
                            "Total Paragraphs": len(justext_paragraphs),
                            "Content Paragraphs": len([p for p in justext_paragraphs if not p.is_boilerplate]),
                            "Boilerplate Removed": len([p for p in justext_paragraphs if p.is_boilerplate])
                        }

                        display_library_results("JusText", title, date, justext_text, pdf_links, has_pdf,
                                                additional_info)

                        results['JusText'] = {
                            'title': title,
                            'date': date,
                            'content_length': len(justext_text) if justext_text else 0,
                            'has_metadata': title != "Not found"
                        }

                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ JusText error: {str(e)}</div>', unsafe_allow_html=True)
                        results['JusText'] = {'error': str(e)}

                # 3. HTML2Text
                with tab3:
                    try:
                        h = html2text.HTML2Text()
                        h.ignore_links = False
                        h.body_width = 0  # No line wrapping
                        html2text_content = h.handle(html_content)

                        title, date = extract_basic_metadata(html_content)

                        additional_info = {
                            "Links Preserved": "Yes",
                            "Markdown Format": "Yes",
                            "Line Wrapping": "Disabled"
                        }

                        display_library_results("HTML2Text", title, date, html2text_content, pdf_links, has_pdf,
                                                additional_info)

                        results['HTML2Text'] = {
                            'title': title,
                            'date': date,
                            'content_length': len(html2text_content) if html2text_content else 0,
                            'has_metadata': title != "Not found"
                        }

                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ HTML2Text error: {str(e)}</div>', unsafe_allow_html=True)
                        results['HTML2Text'] = {'error': str(e)}

                # 4. Inscriptis
                with tab4:
                    try:
                        inscriptis_text = inscriptis.get_text(html_content)

                        title, date = extract_basic_metadata(html_content)

                        additional_info = {
                            "Layout Preservation": "Yes",
                            "Table Support": "Yes",
                            "CSS Awareness": "Basic"
                        }

                        display_library_results("Inscriptis", title, date, inscriptis_text, pdf_links, has_pdf,
                                                additional_info)

                        results['Inscriptis'] = {
                            'title': title,
                            'date': date,
                            'content_length': len(inscriptis_text) if inscriptis_text else 0,
                            'has_metadata': title != "Not found"
                        }

                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ Inscriptis error: {str(e)}</div>',
                                    unsafe_allow_html=True)
                        results['Inscriptis'] = {'error': str(e)}

                # 5. BeautifulSoup
                with tab5:
                    try:
                        soup = BeautifulSoup(html_content, 'html.parser')

                        # Remove unwanted elements
                        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                            element.decompose()

                        bs_text = soup.get_text(separator='\n', strip=True)
                        title, date = extract_basic_metadata(html_content)

                        additional_info = {
                            "Custom Extraction": "Yes",
                            "Element Filtering": "script, style, nav, footer, header removed",
                            "Flexibility": "High"
                        }

                        display_library_results("BeautifulSoup", title, date, bs_text, pdf_links, has_pdf,
                                                additional_info)

                        results['BeautifulSoup'] = {
                            'title': title,
                            'date': date,
                            'content_length': len(bs_text) if bs_text else 0,
                            'has_metadata': title != "Not found"
                        }

                    except Exception as e:
                        st.markdown(f'<div class="error-box">❌ BeautifulSoup error: {str(e)}</div>',
                                    unsafe_allow_html=True)
                        results['BeautifulSoup'] = {'error': str(e)}

                # 6. Summary
                with tab6:
                    st.markdown('<div class="library-header">📊 Comparison Summary</div>', unsafe_allow_html=True)

                    # Create comparison dataframe
                    comparison_data = []
                    for lib_name, lib_results in results.items():
                        if 'error' not in lib_results:
                            comparison_data.append({
                                'Library': lib_name,
                                'Title Found': '✅' if lib_results.get('title', 'Not found') != 'Not found' else '❌',
                                'Date Found': '✅' if lib_results.get('date', 'Not found') != 'Not found' else '❌',
                                'Content Length': f"{lib_results.get('content_length', 0):,}",
                                'Has Metadata': '✅' if lib_results.get('has_metadata', False) else '❌'
                            })

                    if comparison_data:
                        df = pd.DataFrame(comparison_data)
                        st.dataframe(df, use_container_width=True)

                    # PDF Summary
                    st.markdown('<div class="section-title">📎 PDF Detection Summary</div>', unsafe_allow_html=True)
                    if has_pdf:
                        st.markdown(f'<div class="success-box">✅ Found {len(pdf_links)} PDF link(s) on this page</div>',
                                    unsafe_allow_html=True)
                        for i, pdf in enumerate(pdf_links, 1):
                            st.markdown(
                                f'<div class="info-box"><strong>PDF {i}:</strong> <a href="{pdf["url"]}" target="_blank">{pdf["text"][:50]}...</a></div>',
                                unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">❌ No PDF links found on this page</div>',
                                    unsafe_allow_html=True)

                    # Recommendations
                    st.markdown('<div class="section-title">💡 Recommendations</div>', unsafe_allow_html=True)

                    best_for_content = max(results.items(),
                                           key=lambda x: x[1].get('content_length', 0) if 'error' not in x[1] else 0)
                    best_for_metadata = next(
                        (lib for lib, res in results.items() if res.get('has_metadata', False) and 'error' not in res),
                        None)

                    recommendations = []
                    if best_for_content[1].get('content_length', 0) > 0:
                        recommendations.append(
                            f"<strong>Best for content extraction:</strong> {best_for_content[0]} ({best_for_content[1]['content_length']:,} characters)")

                    if best_for_metadata:
                        recommendations.append(
                            f"<strong>Best for metadata:</strong> {best_for_metadata} (successfully extracted title/date)")

                    recommendations.extend([
                        "<strong>For news articles:</strong> Trafilatura (specialized for article content)",
                        "<strong>For preserving links:</strong> HTML2Text (maintains markdown links)",
                        "<strong>For layout preservation:</strong> Inscriptis (maintains visual structure)",
                        "<strong>For custom extraction:</strong> BeautifulSoup (maximum flexibility)"
                    ])

                    rec_text = "<br>• ".join(recommendations)
                    st.markdown(f'<div class="info-box">• {rec_text}</div>', unsafe_allow_html=True)

        except requests.exceptions.RequestException as e:
            st.markdown(f'<div class="error-box">❌ Error fetching URL: {str(e)}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error-box">❌ Unexpected error: {str(e)}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

