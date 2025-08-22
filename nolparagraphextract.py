import os
import re
import logging
import pandas as pd
import html
from typing import Dict, List, Tuple, Optional, Set
from bs4 import BeautifulSoup
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class Config:
    human_csv_path: str = ""
    raw_text_dir: str = ""
    output_path: str = "output"
    threshold: float = 0.2
    max_paragraph_words: int = 150
    max_total_words: int = 300
    max_candidates: int = 2
    min_paragraph_words: int = 20

@dataclass
class NOLCandidate:
    text: str
    score: float
    start_pos: int
    end_pos: int
    section_header: str = ""
    unit_info: str = ""
    paragraph_index: int = 0
    has_numbers: bool = False
    nol_amount_count: int = 0
    specific_nol_terms: Set[str] = None
    word_count: int = 0
    source_tag: str = ""

class NOLExtractor:
    def __init__(self, config: Config):
        self.config = config
        self._compile_patterns()
        
    def _compile_patterns(self):
        basic_nol_terms = [
            r'net\s+operating\s+loss(?:es)?',
            r'\bNOL\b',
            r'loss\s+carry(?:forward|back)s?',
            r'carry(?:forward|back)\s+(?:of\s+)?(?:net\s+operating\s+)?loss(?:es)?',
            r'operating\s+loss\s+carry(?:forward|back)s?',
            r'tax\s+loss\s+carry(?:forward|back)s?'
        ]
        
        boosted_nol_terms = [
            r'(?:federal|state|foreign|domestic|U\.?S\.?)\s+(?:NOL|net\s+operating\s+loss)',
            r'(?:NOL|net\s+operating\s+loss)\s+carryforward\s+(?:available|balance)',
            r'(?:NOL|net\s+operating\s+loss)\s+expir(?:e|ing|ation)',
            r'valuation\s+allowance.*(?:NOL|net\s+operating\s+loss)',
            r'(?:NOL|net\s+operating\s+loss).*valuation\s+allowance'
        ]
        
        amount_indicators = [
            r'(?:NOL|net\s+operating\s+loss).*(?:\$|million|billion|thousand)',
            r'(?:\$|million|billion|thousand).*(?:NOL|net\s+operating\s+loss)',
            r'(?:NOL|net\s+operating\s+loss)\s+of\s+(?:\$|approximately|\d)',
            r'(?:NOL|net\s+operating\s+loss).*(?:totaling?|amount(?:ing|ed)?)\s+to\s+(?:\$|\d)',
            r'(?:\$|\d).*(?:NOL|net\s+operating\s+loss).*(?:expire|expiring|expiration)'
        ]
        
        unit_patterns = [
            r'(?:in\s+)?(?:thousands?|millions?)(?:\s+of\s+dollars?)?',
            r'amounts?\s+in\s+(?:thousands?|millions?)',
            r'\(\$?\s*in\s+(?:thousands?|millions?)\s*\)',
            r'\(000s?\)',
            r'\(000,000s?\)'
        ]
        
        section_headers = [
            r'notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements?',
            r'income\s+taxes?',
            r'tax(?:es|ation)',
            r'deferred\s+tax',
            r'note\s+\d+.*(?:tax|income)'
        ]
        
        self.basic_patterns = [re.compile(t, re.IGNORECASE) for t in basic_nol_terms]
        self.boosted_patterns = [re.compile(t, re.IGNORECASE) for t in boosted_nol_terms]
        self.amount_patterns = [re.compile(t, re.IGNORECASE) for t in amount_indicators]
        self.unit_patterns = [re.compile(p, re.IGNORECASE) for p in unit_patterns]
        self.section_patterns = [re.compile(p, re.IGNORECASE) for p in section_headers]
        
        self.dollar_pattern = re.compile(r'\$\s*[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand|m|b|k))?', re.IGNORECASE)
        self.number_pattern = re.compile(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b')
        
        self.unwanted_patterns = [
            re.compile(r'^\s*[\d\.\s,-]+\s*$'),
            re.compile(r'^\s*(?:[A-Z]\d{1,3}(?:[A-Z])?|\d{1,3}(?:[A-Z])?)\s*$', re.IGNORECASE),
            re.compile(r'\b(?:ITEM|FORM)\s+\d+(?:[A-Z])?\b', re.IGNORECASE),
            re.compile(r'(?:exhibit\s+)?index|(?:table\s+of\s+)?contents?', re.IGNORECASE),
            re.compile(r'^\s*\(cid:\d+\)\s*$'),
            re.compile(r'^\s*&#\d+;\s*$')
        ]

    def _clean_text(self, text: str) -> str:
        text = html.unescape(text)
        text = re.sub(r'\s+', ' ', text).strip()
        if any(p.fullmatch(text) for p in self.unwanted_patterns):
            return ""
        return text

    def split_text(self, text: str) -> List[str]:
        text = self._clean_text(text)
        if not text:
            return []
        
        words = text.split()
        if self.config.min_paragraph_words <= len(words) <= self.config.max_paragraph_words:
            return [text]
        
        paragraphs = []
        sections = text.split('\n\n')
        
        for section in sections:
            section = self._clean_text(section)
            if not section:
                continue
            
            words = section.split()
            if self.config.min_paragraph_words <= len(words) <= self.config.max_paragraph_words:
                paragraphs.append(section)
            else:
                sentences = re.split(r'(?<=[.!?])\s+', section)
                current = []
                current_count = 0
                
                for sentence in sentences:
                    sentence = self._clean_text(sentence)
                    if not sentence:
                        continue
                    
                    sentence_words = sentence.split()
                    if current_count + len(sentence_words) > self.config.max_paragraph_words:
                        if current and self.config.min_paragraph_words <= current_count:
                            paragraphs.append(' '.join(current))
                        current = [sentence]
                        current_count = len(sentence_words)
                    else:
                        current.append(sentence)
                        current_count += len(sentence_words)
                
                if current and self.config.min_paragraph_words <= current_count:
                    paragraphs.append(' '.join(current))
        
        return paragraphs

    def extract_structure(self, html_content: str) -> Tuple[str, List[Tuple[str, str]], Dict[str, str]]:
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for element in soup(["script", "style", "comment", "sup", "sub"]):
                element.decompose()
            
            for element in soup.find_all(lambda tag: tag.has_attr('style') and 'display:none' in tag['style'].replace(" ", "")):
                element.decompose()
            
            paragraphs_with_tags = []
            
            for tag in soup.find_all(['p', 'div', 'td', 'li']):
                if tag.name == 'td' and tag.find('table'):
                    continue
                
                text = tag.get_text(separator=' ', strip=True)
                split_paragraphs = self.split_text(text)
                
                for sp in split_paragraphs:
                    cleaned = self._clean_text(sp)
                    if cleaned:
                        paragraphs_with_tags.append((cleaned, tag.name))
            
            unique_paragraphs = []
            seen = set()
            
            for text, tag in paragraphs_with_tags:
                key = text[:200].lower()
                if key not in seen:
                    seen.add(key)
                    word_count = len(text.split())
                    if self.config.min_paragraph_words <= word_count <= self.config.max_paragraph_words:
                        unique_paragraphs.append((text, tag))
            
            full_text = soup.get_text(separator=' ', strip=True)
            full_text = self._clean_text(full_text)
            
            sections = {}
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']):
                header_text = self._clean_text(tag.get_text(strip=True))
                if header_text and any(p.search(header_text) for p in self.section_patterns):
                    sections[header_text] = header_text
            
            return full_text, unique_paragraphs, sections
            
        except Exception as e:
            logging.error(f"Error extracting structured text from HTML: {e}")
            return "", [], {}

    def detect_units(self, html_content: str, bs_tag) -> str:
        units = set()
        
        if bs_tag:
            text = bs_tag.get_text(separator=' ', strip=True)
            for pattern in self.unit_patterns:
                match = pattern.search(text)
                if match:
                    units.add(match.group(0))
            
            for parent in bs_tag.parents:
                if parent.name in ['table', 'div', 'body']:
                    parent_text = parent.get_text(separator=' ', strip=True)
                    for pattern in self.unit_patterns:
                        match = pattern.search(parent_text)
                        if match:
                            units.add(f"From {parent.name}: {match.group(0)}")
                    if parent.name == 'body':
                        break
        
        return "; ".join(sorted(units))

    def score_paragraph(self, text: str, index: int, section: str = "", units: str = "", tag: str = "p") -> NOLCandidate:
        word_count = len(text.split())
        
        if not (self.config.min_paragraph_words <= word_count <= self.config.max_paragraph_words):
            return NOLCandidate(text=text, score=0.0, start_pos=0, end_pos=len(text), 
                              word_count=word_count, source_tag=tag)
        
        score = 0.0
        terms = set()
        
        for pattern in self.basic_patterns:
            matches = pattern.findall(text)
            if matches:
                score += len(matches) * 2.0
                terms.update(m.lower() for m in matches)
        
        for pattern in self.boosted_patterns:
            matches = pattern.findall(text)
            if matches:
                score += len(matches) * 5.0
                terms.update(m.lower() for m in matches)
        
        for pattern in self.amount_patterns:
            matches = pattern.findall(text)
            if matches:
                score += len(matches) * 8.0
                terms.update(m.lower() for m in matches)
        
        dollar_matches = self.dollar_pattern.findall(text)
        number_matches = self.number_pattern.findall(text)
        has_numbers = len(dollar_matches) > 0 or len(number_matches) > 0
        
        if has_numbers:
            score += 10.0 if number_matches else 0
            score += 15.0 if dollar_matches else 0
        elif score > 0:
            score *= 0.3
        
        nol_amount_count = 0
        nol_amount_patterns = [
            r'(?:NOL|net\s+operating\s+loss).*\$[\d,]+',
            r'\$[\d,]+.*(?:NOL|net\s+operating\s+loss)',
            r'(?:NOL|net\s+operating\s+loss).*\d[\d,]*(?:\.\d+)?(?:k|m|b|thousand|million|billion)?'
        ]
        
        for pattern_str in nol_amount_patterns:
            pattern = re.compile(pattern_str, re.IGNORECASE)
            nol_amount_count += len(pattern.findall(text))
        
        score += nol_amount_count * 12.0
        
        if section and any(p.search(section) for p in self.section_patterns):
            score += 3.0
        
        if units:
            score += 2.0
        
        if tag == 'td':
            score *= 0.85
        elif tag == 'p':
            score *= 1.05
        
        if word_count > 100:
            score *= 0.95
        elif word_count < 30:
            score *= 0.9
        
        score = max(0, score - index * 0.005)
        
        return NOLCandidate(
            text=text, score=score, start_pos=0, end_pos=len(text),
            section_header=section, unit_info=units, paragraph_index=index,
            has_numbers=has_numbers, nol_amount_count=nol_amount_count,
            specific_nol_terms=terms, word_count=word_count, source_tag=tag
        )

    def extract_candidates(self, html_content: str) -> List[NOLCandidate]:
        full_text, paragraphs_with_tags, sections = self.extract_structure(html_content)
        
        if not paragraphs_with_tags:
            return []
        
        candidates = []
        soup = BeautifulSoup(html_content, 'html.parser')
        all_tags = soup.find_all(['p', 'div', 'td', 'li'])
        
        tag_map = {}
        for tag in all_tags:
            cleaned = self._clean_text(tag.get_text(separator=' ', strip=True))
            if cleaned:
                tag_map[cleaned] = tag
        
        for i, (text, source_tag) in enumerate(paragraphs_with_tags):
            word_count = len(text.split())
            if not (self.config.min_paragraph_words <= word_count <= self.config.max_paragraph_words):
                continue
            
            has_nol = any(p.search(text) for p in self.basic_patterns + self.boosted_patterns)
            
            if has_nol:
                original_tag = tag_map.get(text)
                if not original_tag:
                    try:
                        found = soup.find(lambda t: text in self._clean_text(t.get_text(separator=' ', strip=True)))
                        if found:
                            original_tag = found
                    except Exception:
                        pass
                
                units = self.detect_units(html_content, original_tag) if original_tag else ""
                
                section = ""
                for header in sections.keys():
                    if any(p.search(header) for p in self.section_patterns):
                        section = header
                        break
                
                candidate = self.score_paragraph(text, i, section, units, source_tag)
                
                if candidate.score > 1.0:
                    candidates.append(candidate)
        
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def extract_nol(self, html_content: str) -> Tuple[str, float, Dict]:
        candidates = self.extract_candidates(html_content)
        
        if not candidates:
            return "", 0.0, {}
        
        selected = []
        with_numbers = [c for c in candidates if c.has_numbers]
        
        for c in with_numbers[:self.config.max_candidates]:
            selected.append(c)
        
        if len(selected) < self.config.max_candidates:
            for c in candidates:
                if c not in selected and len(selected) < self.config.max_candidates:
                    selected.append(c)
        
        selected.sort(key=lambda x: x.paragraph_index)
        
        texts = []
        total_words = 0
        total_score = 0
        units = set()
        sections = set()
        truncated = False
        
        for candidate in selected:
            if total_words + candidate.word_count <= self.config.max_total_words:
                texts.append(candidate.text)
                total_words += candidate.word_count
                total_score += candidate.score
                
                if candidate.unit_info:
                    units.add(candidate.unit_info)
                if candidate.section_header:
                    sections.add(candidate.section_header)
            else:
                remaining = self.config.max_total_words - total_words
                if remaining >= self.config.min_paragraph_words:
                    words = candidate.text.split()
                    truncated_text = ' '.join(words[:remaining]) + "..."
                    texts.append(truncated_text)
                    total_words += len(truncated_text.split())
                    total_score += candidate.score * 0.5
                    truncated = True
                else:
                    truncated = True
                break
        
        combined = "\n\n".join(texts)
        
        words_list = combined.split()
        if len(words_list) > self.config.max_total_words:
            combined = ' '.join(words_list[:self.config.max_total_words]) + "..."
            truncated = True
        
        confidence = min(1.0, total_score / 50.0)
        
        metadata = {
            'num_candidates_considered': len(candidates),
            'num_candidates_selected': len(selected),
            'total_score': total_score,
            'confidence': confidence,
            'unit_info': list(units),
            'section_headers': list(sections),
            'has_numbers': any(c.has_numbers for c in selected),
            'nol_amount_count': sum(c.nol_amount_count for c in selected),
            'final_word_count': len(combined.split()),
            'was_truncated': truncated
        }
        
        return combined, confidence, metadata

    def process_file(self, file_path: str) -> Dict:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
            
            nol_text, confidence, metadata = self.extract_nol(html_content)
            
            file_name = os.path.basename(file_path)
            parts = file_name.split('-')
            cik = parts[0] if len(parts) > 0 else ""
            date_str = parts[1].split('.')[0] if len(parts) > 1 else ""
            
            has_nol = confidence > self.config.threshold and metadata.get('has_numbers', False)
            
            return {
                'file_path': file_path,
                'file_name': file_name,
                'cik': cik,
                'date': date_str,
                'has_nol': has_nol,
                'nol_text': nol_text if has_nol else '',
                'confidence': confidence,
                'word_count': metadata.get('final_word_count', 0),
                'num_candidates_selected': metadata.get('num_candidates_selected', 0),
                'has_numbers': metadata.get('has_numbers', False),
                'nol_amount_count': metadata.get('nol_amount_count', 0),
                'was_truncated': metadata.get('was_truncated', False),
                'unit_info': '; '.join(metadata.get('unit_info', [])),
                'section_headers': '; '.join(metadata.get('section_headers', []))
            }
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'cik': '',
                'date': '',
                'has_nol': False,
                'nol_text': '',
                'confidence': 0.0,
                'word_count': 0,
                'num_candidates_selected': 0,
                'has_numbers': False,
                'nol_amount_count': 0,
                'was_truncated': False,
                'unit_info': '',
                'section_headers': ''
            }

def load_csv(csv_path: str) -> pd.DataFrame:
    encodings = ['utf-8', 'latin-1', 'iso-8859-1']
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, on_bad_lines='warn', encoding=encoding)
            logging.info(f"✅ Successfully loaded CSV with {len(df)} rows using {encoding} encoding")
            return df
        except Exception as e:
            logging.warning(f"Failed to load CSV with {encoding} encoding: {e}")
            continue
    raise ValueError(f"Could not load CSV file: {csv_path} with any tried encoding.")

def compare_results(old_path: str, new_path: str):
    logging.info("\n=== Comparing Old vs New Results ===")
    
    try:
        old_df = load_csv(old_path)
        new_df = load_csv(new_path)
        
        old_df['Year'] = old_df['date'].astype(str).str[:4]
        new_df['Year'] = new_df['date'].astype(str).str[:4]
        
        if 'word_count' not in old_df.columns:
            old_df['word_count'] = old_df['nol_text'].apply(
                lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() else 0
            )
        
        logging.info(f"Old results: {len(old_df)} files")
        logging.info(f"New results: {len(new_df)} files")
        
        logging.info(f"\n=== Word Count Comparison ===")
        
        old_with_nol = old_df[old_df['word_count'] > 0]
        new_with_nol = new_df[new_df['word_count'] > 0]
        
        logging.info(f"Old: {len(old_with_nol)} files with NOL text")
        logging.info(f"New: {len(new_with_nol)} files with NOL text")
        
        if len(old_with_nol) > 0 and len(new_with_nol) > 0:
            logging.info(f"\nWord count statistics:")
            logging.info(f"                    Old      New")
            logging.info(f"Average words:   {old_with_nol['word_count'].mean():7.1f}  {new_with_nol['word_count'].mean():7.1f}")
            logging.info(f"Median words:    {old_with_nol['word_count'].median():7.1f}  {new_with_nol['word_count'].median():7.1f}")
            logging.info(f"Max words:       {old_with_nol['word_count'].max():7.0f}  {new_with_nol['word_count'].max():7.0f}")
            logging.info(f"Over 500 words:  {(old_with_nol['word_count'] > 500).sum():7.0f}  {(new_with_nol['word_count'] > 500).sum():7.0f}")
            logging.info(f"Over 1000 words: {(old_with_nol['word_count'] > 1000).sum():7.0f}  {(new_with_nol['word_count'] > 1000).sum():7.0f}")
            
            logging.info(f"\n=== Percentage Over Max Word Limit (300 Words) by Year ===")
            logging.info(f"Year | Old (>500 words) % | New (>300 words) %")
            logging.info(f"-" * 55)
            
            for year in sorted(set(old_df['Year'].unique()) | set(new_df['Year'].unique())):
                old_year_nol = old_df[(old_df['Year'] == year) & (old_df['word_count'] > 0)]
                new_year_nol = new_df[(new_df['Year'] == year) & (new_df['word_count'] > 0)]
                
                old_pct = (old_year_nol['word_count'] > 500).mean() * 100 if len(old_year_nol) > 0 else 0
                new_pct = (new_year_nol['word_count'] > 300).mean() * 100 if len(new_year_nol) > 0 else 0
                
                if old_pct > 0 or new_pct > 0:
                    logging.info(f"{year} | {old_pct:17.1f}% | {new_pct:20.1f}%")
    except Exception as e:
        logging.error(f"Error comparing results: {e}")

def main():
    logging.info("Running Improved NOL Extractor")
    logging.info("=" * 50)
    
    config = Config(
        max_paragraph_words=150,
        max_total_words=300,
        max_candidates=2,
        min_paragraph_words=20,
        threshold=0.2
    )
    
    logging.info(f"Configuration:")
    logging.info(f"  Max paragraph words: {config.max_paragraph_words}")
    logging.info(f"  Max total words: {config.max_total_words}")
    logging.info(f"  Max candidates: {config.max_candidates}")
    logging.info(f"  Min paragraph words: {config.min_paragraph_words}")
    logging.info(f"  Confidence Threshold: {config.threshold}")
    
    extractor = NOLExtractor(config)
    
    oos_dir = "/Users/jingyuanchen/Desktop/econ199/SEC_Filings/html"
    output_dir = "/Users/jingyuanchen/Desktop/nolextractor/insample"
    
    if not os.path.exists(oos_dir):
        logging.error(f"OOS directory not found: {oos_dir}")
        return None
    
    html_files = [f for f in os.listdir(oos_dir) if f.endswith(('.htm', '.html'))]
    logging.info(f"\nFound {len(html_files)} HTML files to process in {oos_dir}")
    
    if not html_files:
        logging.warning("No HTML files found to process!")
        return None
    
    results = []
    processed_count = 0
    
    for filename in html_files:
        file_path = os.path.join(oos_dir, filename)
        results.append(extractor.process_file(file_path))
        
        processed_count += 1
        if processed_count % 100 == 0:
            logging.info(f"Processed {processed_count}/{len(html_files)} files...")
    
    df = pd.DataFrame(results)
    df['Year'] = df['date'].astype(str).str[:4]
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "improved_oos_results.csv")
    df.to_csv(output_path, index=False)
    
    logging.info(f"\n✅ Processing complete!")
    logging.info(f"Results saved to: {output_path}")
    
    logging.info(f"\n=== Results Analysis (Improved Extractor) ===")
    nol_found = df['has_nol'].sum()
    logging.info(f"Total files processed: {len(df)}")
    logging.info(f"NOLs detected (confidence > {config.threshold} and has numbers): {nol_found}")
    logging.info(f"Detection rate: {nol_found/len(df)*100:.1f}%")
    
    if nol_found > 0:
        nol_df = df[df['has_nol']]
        logging.info(f"Average word count for detected NOLs: {nol_df['word_count'].mean():.1f}")
        logging.info(f"Max word count for detected NOLs: {nol_df['word_count'].max()}")
        logging.info(f"Files with NOL text exceeding {config.max_total_words} words (should be few): {(nol_df['word_count'] > config.max_total_words).sum()}")
        logging.info(f"Files where final output was truncated: {nol_df['was_truncated'].sum()}")
        
        logging.info(f"\n=== Year Analysis (Improved Extractor - Truncation) ===")
        
        if nol_found > 0:
            year_analysis = nol_df.groupby('Year').apply(
                lambda x: pd.Series({
                    'Total_NOL_Files': len(x),
                    'Over_Max_Words': (x['word_count'] > config.max_total_words).sum(),
                    'Pct_Over_Max_Words': (x['word_count'] > config.max_total_words).mean() * 100 if len(x) > 0 else 0,
                    'Was_Truncated_Count': x['was_truncated'].sum()
                })
            ).round(1)
            logging.info("NOL Text Length Analysis by Year (for detected NOLs):")
            logging.info(year_analysis.to_string())
        
        logging.info(f"\n=== Sample NOL Extractions ===")
        sample_nols = df[(df['has_nol']) & (df['nol_text'] != '')].head(3)
        
        for i, (_, row) in enumerate(sample_nols.iterrows()):
            logging.info(f"\n--- Example {i+1} ---")
            logging.info(f"File: {row['file_name']}")
            logging.info(f"Year: {row['Year']}")
            logging.info(f"Confidence: {row['confidence']:.3f}")
            logging.info(f"Word count: {row['word_count']}")
            logging.info(f"Was Truncated: {row['was_truncated']}")
            logging.info(f"Unit Info: {row['unit_info'] or 'N/A'}")
            logging.info(f"Section Headers: {row['section_headers'] or 'N/A'}")
            logging.info(f"NOL Text Preview: {row['nol_text'][:200]}...")
    
    old_results = os.path.join(output_dir, "oos_results.csv")
    if os.path.exists(old_results):
        compare_results(old_results, output_path)
    else:
        logging.info(f"\nOld results file not found at {old_results}")
        logging.info("Run the original pipeline first to compare results.")
    
    return df

if __name__ == "__main__":
    df = main()
    
    if df is not None:
        logging.info(f"\n extraction complete see 'improved_oos_results.csv'")
