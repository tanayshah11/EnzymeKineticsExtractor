# 🧬 EnzymeKineticsExtractor

**AI-powered tool for automated extraction of enzyme kinetic parameters from scientific literature**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Google Gemini](https://img.shields.io/badge/AI-Google%20Gemini-green.svg)](https://ai.google.dev/)

## 📋 Overview

EnzymeKineticsExtractor is a comprehensive tool that automatically extracts enzyme kinetic parameters (kcat, Km, kcat/Km) from scientific literature. Using Google Gemini's advanced AI capabilities, it processes thousands of research papers from PubMed Central to build structured datasets for biochemical research.

### 🎯 Key Features

- **📄 Full-Text Processing**: Extracts from PMC full-text articles and PubMed abstracts
- **📊 Table & Figure Analysis**: Intelligently parses tables and figure captions where kinetic data is often found
- **🤖 AI-Powered Extraction**: Uses Google Gemini 2.5 for precise parameter identification
- **⚡ Smart Rate Limiting**: Intelligent handling of API quotas with automatic retry mechanisms
- **🔄 Smart Caching**: Intelligently caches content and results to avoid redundant processing
- **💾 Progress Tracking**: Resume capability with automatic progress saving
- **📈 Comprehensive Logging**: Detailed extraction statistics and error reporting

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key (free tier available)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/tanayshah11/EnzymeKineticsExtractor.git
   cd EnzymeKineticsExtractor
   ```

2. **Create virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API key:**
   - Get your free Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
   - Create a `.env` file:
     ```bash
     GEMINI_API_KEY=your_api_key_here
     ```

### Usage

**Test mode (5 random papers):**

```bash
python extractor.py --test
```

**Process the full dataset:**

```bash
python extractor.py
```

**Custom processing:**

```bash
# Use different model
python extractor.py --model gemini-2.5-pro

# Custom delay between requests
python extractor.py --delay 6.0
```

## 📊 Data Structure

### Input Format

The tool expects a CSV file with these columns:

- `enzyme_name`: Name of the enzyme
- `mutation_type`: Type of mutation (e.g., "A->V")
- `activity_change`: Description of activity change
- `pubmed_link`: URL to the PubMed article

### Output Format

Extracted data includes 12 new columns for each parameter:

- `kcat_value`, `kcat_unit`, `kcat_substrate`, `kcat_notes`
- `km_value`, `km_unit`, `km_substrate`, `km_notes`
- `kcat_km_value`, `kcat_km_unit`, `kcat_km_substrate`, `kcat_km_notes`

Plus metadata:

- `extraction_status`: Success/failure status
- `paper_content_type`: Type of content retrieved (PMC/PubMed)

## 🎛️ Model Options

| Model                      | RPM | RPD   | Context | Best For               |
| -------------------------- | --- | ----- | ------- | ---------------------- |
| `gemini-2.5-flash-lite` ✅ | 15  | 1,000 | 250K    | Large-scale processing |
| `gemini-2.5-flash`         | 10  | 250   | 250K    | Balanced performance   |
| `gemini-2.5-pro`           | 5   | 100   | 250K    | Highest quality        |

## 📁 Project Structure

```
EnzymeKineticsExtractor/
├── extractor.py              # Main extraction script
├── requirements.txt          # Python dependencies
├── .env                      # API key configuration
├── enzyme_mutations_clean.csv # Main dataset (1,985 entries)
└── README.md
```

## 🔧 Advanced Configuration

### Rate Limiting

The tool automatically calculates optimal delays based on:

- **Requests Per Minute (RPM)** limits
- **Tokens Per Minute (TPM)** limits
- Average paper size (~20K tokens)

### Error Handling

- Automatic retry on rate limit errors
- Graceful fallback from PMC to PubMed abstracts
- Progress saving every 10 papers
- Comprehensive error logging

### Smart Caching

The tool automatically caches paper content and extraction results to avoid redundant processing:

- **Content Caching**: Papers are fetched once and cached by PMID
- **Extraction Caching**: Results are cached by content hash + mutation type
- **Efficiency**: Processes only ~180 unique papers for 1,985 dataset entries

## 📈 Performance

**Expected Processing Times:**

- **Test Mode**: ~2-3 minutes (10 papers)
- **Full Dataset**: ~12-15 minutes (1,985 entries, ~180 unique papers)
- **Caching Benefits**: 90% reduction in API calls for duplicate papers

**API Usage (Free Tier):**

- Full Dataset: ~180 unique requests, ~3.6M tokens
- Caching Savings: ~85% reduction vs processing all 1,985 entries
- Processing Time: ~12-15 minutes total

## 🧪 Example Results

```
📈 Extraction Summary:
  - Completed: 650/700
  - Papers with kcat: 234
  - Papers with Km: 198
  - Papers with kcat/Km: 156
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

**Common Issues:**

- **Rate Limit Errors**: Increase `--delay` parameter or use a model with higher limits
- **No PMC Access**: Tool automatically falls back to PubMed abstracts
- **API Key Issues**: Ensure `.env` file is in the project root with valid key

## 📚 Scientific Applications

This tool is designed for:

- **Enzyme Engineering**: Building mutation effect databases
- **Drug Discovery**: Analyzing target enzyme kinetics
- **Computational Biology**: Training machine learning models
- **Systems Biology**: Large-scale kinetic parameter compilation
- **Biochemical Research**: Literature meta-analysis

## 📄 Citation

If you use this tool in your research, please cite:

```bibtex
@software{shah2024enzymekineticsextractor,
  title={EnzymeKineticsExtractor: AI-powered extraction of enzyme kinetic parameters from scientific literature},
  author={Shah, Tanay},
  year={2024},
  url={https://github.com/tanayshah11/EnzymeKineticsExtractor}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini API** for advanced language processing capabilities
- **PubMed Central** for providing open access to scientific literature
- **NCBI** for comprehensive biomedical databases
- **Research Community** for making scientific knowledge accessible

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/tanayshah11/EnzymeKineticsExtractor/issues)
- 💡 **Feature Requests**: [GitHub Discussions](https://github.com/tanayshah11/EnzymeKineticsExtractor/discussions)

---

**Made with ❤️ for the scientific community**

_Empowering biochemical research through automated literature mining_
