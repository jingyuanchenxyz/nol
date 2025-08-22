


### download10k.py
Downloads 10ks from the SEC website based off of a csv or xlsx, works well, make sure format is as follows:
```

```


Downloads files as:
```

```


### nolparagraphextract.py
Extracts candidate NOL paragraphs for LLM to parse later, main issue: units retainment

CONFIG
Controls extraction behavior with the following params:

threshold: min confidence score (0.2 default) for NOL detection
max_paragraph_words: max words per paragraph (150)
max_total_words: max total output words (300)
max_candidates: max number of paragraphs to combine (2)
min_paragraph_words: min words to consider valid (20)

Thresholds
- Score > 1.0: Minimum to consider paragraph
- Score > threshold (0.2): Combined with has_numbers for final detection
- Confidence = min(1.0, total_score / 50.0)

Word Limits
- Enforced at paragraph level (20-150 words)
- Enforced at output level (max 300 words total)
- Truncation applied if limits exceeded

Paths
- Input: `/Users/jingyuanchen/Desktop/econ199/SEC_Filings/html`
- Output: `/Users/jingyuanchen/Desktop/nolextractor/insample/improved_oos_results.csv`

Change Config:

```python
config = Config(
    max_paragraph_words=150,
    max_total_words=300,
    threshold=0.2
)

extractor = NOLExtractor(config)
result = extractor.process_file('path/to/10k.html')

```
NOLCandidate
Structure holding extracted paragraph information:

text: The actual paragraph text
score: Relevance score based on NOL patterns
has_numbers: Boolean indicating presence of numerical values
word_count: Number of words in the paragraph
unit_info: Detected monetary units (thousands, millions)
section_header: Associated document section

Pattern Detection
The extractor uses three tiers of regex patterns:
Basic Patterns: Common NOL and NOL adjacent terms (eg. NOL, net operating loss, carryforward)
Boosted Patterns: Specific NOL term targets: (eg. Federal, State, Foreign NOL)
Amount Pattern: NOL mention with numeric values around it

Scoring:
Candidate paragraphs are scored based on:
Pattern matches (2-8 points)
Presence of numbers (10-15 points)
Note: Paragraphs without numbers receive a 70% score penalty.
Direct NOL-amount combinations (12 points boosted)
Section header relevance (3 points)
Unit information (2 points)

Extraction Process
HTML Parsing via BeautifulSoup text taken from p, div, td, li tags
Text Cleaning: strips HTML artifacts, whitespace, etc
Paragraph splitting into chunks
Candidate Selection (finds then ranks paragraphs by NOL score as defined before)
Outputs top candidate 

Unit Detection
Scans for monetary units in current paragraph text
Parent HTML tables and divs
Preceding sibling elements
Captions and headers

Common patterns: "in thousands", "in millions", "(000s)", "($M)"

Output
Returns csv with columns:
- `file_name`, `cik`, `date`: File identifiers
- `has_nol`: Boolean detection result
- `nol_text`: Extracted NOL text (if found)
- `confidence`: Score-based confidence (0-1)
- `word_count`: Final text word count
- `has_numbers`: Presence of numerical values
- `was_truncated`: Whether text was cut for length
- `unit_info`: Detected monetary units
- `section_headers`: Associated document sections

