#!/usr/bin/env python3
"""
Enzyme Parameter Extractor
Extracts kinetic parameters (kcat, Km, kcat/Km) from research papers using Google Gemini Pro
"""

import json
import logging
import time
from typing import Dict, Optional, Tuple, Any
import sys
import re
import random
import os
from dotenv import load_dotenv

# Initialize absl logging before importing Google libraries
from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.ERROR)

import pandas as pd
import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
import urllib3
import google.generativeai as genai

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnzymeParameterExtractor:
    """Extract kinetic parameters from research papers using Google Gemini Pro"""

    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        """
        Initialize the extractor with Google Gemini

        Args:
            model_name: Gemini model to use (2.5-flash-lite has 15 RPM, 250K context, 1000 RPD)
        """
        self.model_name = model_name
        self.ua = UserAgent()
        self.session = requests.Session()
        self.session.verify = False  # Handle SSL issues

        # Initialize Gemini
        self._initialize_gemini()

        # Parameter columns to add
        self.param_columns = {
            "kcat": ["kcat_value", "kcat_unit", "kcat_substrate", "kcat_notes"],
            "km": ["km_value", "km_unit", "km_substrate", "km_notes"],
            "kcat_km": [
                "kcat_km_value",
                "kcat_km_unit",
                "kcat_km_substrate",
                "kcat_km_notes",
            ],
        }

    def _initialize_gemini(self):
        """Initialize Google Gemini with API key from environment"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY not found in .env file")
            logger.error("Please add: GEMINI_API_KEY=your_key_here to .env file")
            sys.exit(1)

        genai.configure(api_key=api_key)

        # Initialize the model with optimal settings
        generation_config = genai.GenerationConfig(
            temperature=0.1,  # Low for precise extraction
            top_p=0.1,
            top_k=10,
            max_output_tokens=2048,
        )

        self.model = genai.GenerativeModel(
            model_name=self.model_name, generation_config=generation_config
        )

        # Test connection silently
        try:
            _ = self.model.generate_content("test")
            logger.info(f"✅ Connected to Gemini {self.model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Gemini: {e}")
            logger.error("Please check your API key and internet connection")
            sys.exit(1)

    def _get_pubmed_id(self, url: str) -> Optional[str]:
        """Extract PubMed ID from URL"""
        if pd.isna(url) or not url:
            return None

        # Extract PMID from various URL formats
        patterns = [
            r"pubmed/(\d+)",
            r"pmid=(\d+)",
            r"PMID:?\s*(\d+)",
            r"(\d{7,8})$",  # Just the ID
        ]

        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _fetch_pmc_full_text(self, pmid: str) -> Optional[Tuple[str, str]]:
        """
        Fetch full text from PubMed Central

        Returns:
            Tuple of (content, content_type) or None
        """
        try:
            # First, check if PMC full text is available
            pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={pmid}&format=json"
            headers = {"User-Agent": self.ua.random}

            response = self.session.get(pmc_url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                records = data.get("records", [])

                if records and "pmcid" in records[0]:
                    pmcid = records[0]["pmcid"]

                    # Fetch PMC full text
                    full_text_url = (
                        f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/"
                    )
                    response = self.session.get(
                        full_text_url, headers=headers, timeout=15
                    )

                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, "html.parser")

                        # Extract main article content
                        article = (
                            soup.find("div", {"class": "jig-ncbiinpagenav"})
                            or soup.find("div", {"id": "maincontent"})
                            or soup.find("article")
                        )

                        if article:
                            # Remove script and style elements
                            for element in article(["script", "style", "nav", "aside"]):
                                element.decompose()

                            content_parts = []

                            # Extract main text
                            main_text = article.get_text(separator="\n", strip=True)
                            content_parts.append("MAIN TEXT:\n" + main_text)

                            # Extract all tables with their captions
                            tables = soup.find_all("table")
                            if tables:
                                content_parts.append("\n\nTABLES FOUND IN PAPER:")
                                for i, table in enumerate(tables, 1):
                                    # Get table caption
                                    caption = None
                                    caption_elem = (
                                        table.find_previous(
                                            "div", class_="table-wrap-title"
                                        )
                                        or table.find_previous("p", class_="caption")
                                        or table.find_previous("caption")
                                    )
                                    if caption_elem:
                                        caption = caption_elem.get_text(strip=True)

                                    content_parts.append(f"\n--- Table {i} ---")
                                    if caption:
                                        content_parts.append(f"Caption: {caption}")

                                    # Extract table data
                                    rows = table.find_all("tr")
                                    table_data = []
                                    for row in rows:
                                        cells = row.find_all(["td", "th"])
                                        row_data = [
                                            cell.get_text(strip=True) for cell in cells
                                        ]
                                        if row_data:
                                            table_data.append(" | ".join(row_data))

                                    if table_data:
                                        content_parts.append("\n".join(table_data))

                            # Extract figure captions (often contain data)
                            figure_captions = soup.find_all(
                                ["figcaption", "div"], class_=["fig-caption", "caption"]
                            )
                            if figure_captions:
                                content_parts.append("\n\nFIGURE CAPTIONS:")
                                for i, caption in enumerate(figure_captions, 1):
                                    caption_text = caption.get_text(strip=True)
                                    if caption_text and len(caption_text) > 20:
                                        content_parts.append(
                                            f"\nFigure {i}: {caption_text}"
                                        )

                            # Look for supplementary data mentions
                            supp_data = soup.find_all(
                                string=re.compile(
                                    r"(Table S\d+|Figure S\d+|Supplementary|Supporting Information)",
                                    re.I,
                                )
                            )
                            if supp_data:
                                content_parts.append(
                                    "\n\nSUPPLEMENTARY DATA REFERENCES FOUND"
                                )
                                for ref in supp_data[
                                    :10
                                ]:  # Limit to first 10 references
                                    if len(ref.strip()) > 10:
                                        content_parts.append(ref.strip()[:200])

                            full_content = "\n".join(content_parts)
                            if (
                                len(full_content) > 500
                            ):  # Ensure we have substantial content
                                logger.info(
                                    f"📄 Retrieved PMC full text with tables for PMID {pmid} ({len(full_content)} chars)"
                                )
                                return full_content, "pmc_full_text"

        except Exception as e:
            logger.debug(f"PMC fetch failed for {pmid}: {e}")

        return None

    def _fetch_pubmed_comprehensive(self, pmid: str) -> Optional[Tuple[str, str]]:
        """
        Fetch comprehensive content from PubMed page

        Returns:
            Tuple of (content, content_type) or None
        """
        try:
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            headers = {"User-Agent": self.ua.random}

            response = self.session.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")

                content_parts = []

                # Extract title
                title_elem = soup.find("h1", {"class": "heading-title"}) or soup.find(
                    "title"
                )
                if title_elem:
                    content_parts.append(f"Title: {title_elem.get_text(strip=True)}")

                # Extract authors
                authors_elem = soup.find("div", {"class": "authors"})
                if authors_elem:
                    content_parts.append(
                        f"Authors: {authors_elem.get_text(strip=True)}"
                    )

                # Extract journal info
                journal_elem = soup.find("button", {"id": "full-view-journal-trigger"})
                if journal_elem:
                    content_parts.append(
                        f"Journal: {journal_elem.get_text(strip=True)}"
                    )

                # Extract abstract - try multiple selectors
                abstract_elem = (
                    soup.find("div", {"class": "abstract-content"})
                    or soup.find("div", {"id": "abstract"})
                    or soup.find("div", {"class": "abstract"})
                    or soup.find("abstract")
                )

                if abstract_elem:
                    abstract_text = abstract_elem.get_text(separator="\n", strip=True)
                    content_parts.append(f"Abstract:\n{abstract_text}")

                # Extract MeSH terms for additional context
                mesh_elem = soup.find("div", {"class": "mesh-terms"})
                if mesh_elem:
                    mesh_text = mesh_elem.get_text(strip=True)
                    content_parts.append(f"MeSH Terms: {mesh_text}")

                # Extract keywords
                keywords_elem = soup.find("div", {"class": "keywords"})
                if keywords_elem:
                    keywords_text = keywords_elem.get_text(strip=True)
                    content_parts.append(f"Keywords: {keywords_text}")

                if content_parts:
                    full_content = "\n\n".join(content_parts)
                    logger.info(
                        f"📝 Retrieved comprehensive PubMed content for PMID {pmid} ({len(full_content)} chars)"
                    )
                    return full_content, "pubmed_comprehensive"

        except Exception as e:
            logger.debug(f"PubMed comprehensive fetch failed for {pmid}: {e}")

        return None

    def fetch_paper_content(self, pubmed_url: str) -> Tuple[Optional[str], str]:
        """
        Fetch paper content, trying PMC full text first, then abstract

        Returns:
            Tuple of (content, content_type)
        """
        pmid = self._get_pubmed_id(pubmed_url)
        if not pmid:
            logger.warning(f"Could not extract PMID from: {pubmed_url}")
            return None, "invalid_url"

        # Try PMC full text first
        result = self._fetch_pmc_full_text(pmid)
        if result:
            return result

        # Fallback to comprehensive PubMed content
        result = self._fetch_pubmed_comprehensive(pmid)
        if result:
            return result

        logger.warning(f"Could not fetch any content for PMID: {pmid}")
        return None, "fetch_failed"

    def extract_parameters_with_gemini(
        self, paper_content: str, enzyme_info: Dict[str, Any]
    ) -> Dict:
        """
        Use Google Gemini to extract kinetic parameters from paper content

        Args:
            paper_content: Text content from the paper
            enzyme_info: Dictionary with enzyme context (name, mutation, etc.)

        Returns:
            Dictionary with extracted parameters
        """
        if not paper_content:
            return self._empty_parameters()

        logger.info(
            f"🧠 Processing {len(paper_content)} characters with Gemini {self.model_name}"
        )

        # Create the prompt
        prompt = self._create_gemini_prompt(paper_content, enzyme_info)

        try:
            # Generate content with Gemini
            response = self.model.generate_content(prompt)

            if response.text:
                logger.debug(f"Gemini response length: {len(response.text)} chars")

                # Extract JSON from response
                json_match = re.search(r"\{.*\}", response.text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group())
                        logger.debug(
                            f"Successfully parsed JSON with {len(result)} keys"
                        )
                        return self._validate_and_format_parameters(result)
                    except json.JSONDecodeError as e:
                        logger.debug(f"JSON decode failed: {e}")
                        logger.debug(f"Response text: {response.text[:500]}")

        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            if "quota" in str(e).lower() or "429" in str(e):
                # Parse retry delay from error message if available
                retry_delay = 60  # Default
                import re
                delay_match = re.search(r'retry in (\d+(?:\.\d+)?)', str(e))
                if delay_match:
                    retry_delay = float(delay_match.group(1)) + 5  # Add 5s buffer

                logger.warning(f"⚠️ Rate limit exceeded. Waiting {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)

                # Try again once after waiting
                try:
                    response = self.model.generate_content(prompt)
                    if response.text:
                        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group())
                            return self._validate_and_format_parameters(result)
                except:
                    logger.warning("⚠️ Retry also failed, skipping this paper")

        return self._empty_parameters()

    def _create_gemini_prompt(self, paper_content: str, enzyme_info: Dict) -> str:
        """Create optimized prompt for Gemini extraction"""

        enzyme_context = f"""
TARGET ENZYME: {enzyme_info.get('enzyme_name', 'Unknown')}
MUTATION: {enzyme_info.get('mutation_type', 'Unknown')}
ACTIVITY CHANGE: {enzyme_info.get('activity_change', 'Unknown')}
        """

        prompt = f"""You are an expert biochemist extracting enzyme kinetic parameters from research papers.

{enzyme_context}

TASK: Extract kinetic parameters (kcat, Km, kcat/Km) for the specific enzyme and mutation above.

SEARCH STRATEGY:
1. FIRST CHECK ALL TABLES - kinetic parameters are often in tables
2. Look for "Table" sections with headers like "Kinetic parameters", "Enzyme activity", "Catalytic constants"
3. Search for numerical values with units: s⁻¹, min⁻¹, h⁻¹, µM, mM, nM, M⁻¹s⁻¹
4. Check figure captions - they often report key values
5. Look for terms: "kcat", "Km", "turnover number", "Michaelis constant", "catalytic efficiency"
6. Focus on data for mutation: {enzyme_info.get('mutation_type', 'Unknown')}

PAPER CONTENT:
{paper_content}

EXTRACTION REQUIREMENTS:
- Extract ONLY explicitly stated numerical values
- Include substrate names when mentioned
- Note experimental conditions (pH, temperature)
- Distinguish mutant from wild-type data
- Use null for missing values

Return ONLY a JSON object with this exact structure (no other text):
{{
    "kcat": {{
        "value": null or number,
        "unit": null or string (e.g., "s⁻¹"),
        "substrate": null or string,
        "notes": null or string
    }},
    "km": {{
        "value": null or number,
        "unit": null or string (e.g., "mM", "µM"),
        "substrate": null or string,
        "notes": null or string
    }},
    "kcat_km": {{
        "value": null or number,
        "unit": null or string (e.g., "M⁻¹s⁻¹"),
        "substrate": null or string,
        "notes": null or string
    }}
}}
"""

        return prompt

    def _validate_and_format_parameters(self, params: Dict) -> Dict:
        """Validate and format extracted parameters"""
        formatted = {}

        for param_type in ["kcat", "km", "kcat_km"]:
            if param_type in params and isinstance(params[param_type], dict):
                param_data = params[param_type]

                # Format the parameter name for column prefixes
                prefix = param_type.replace("/", "_")

                formatted[f"{prefix}_value"] = param_data.get("value")
                formatted[f"{prefix}_unit"] = param_data.get("unit")
                formatted[f"{prefix}_substrate"] = param_data.get("substrate")
                formatted[f"{prefix}_notes"] = param_data.get("notes")
            else:
                # Add empty values
                prefix = param_type.replace("/", "_")
                formatted[f"{prefix}_value"] = None
                formatted[f"{prefix}_unit"] = None
                formatted[f"{prefix}_substrate"] = None
                formatted[f"{prefix}_notes"] = None

        return formatted

    def _empty_parameters(self) -> Dict:
        """Return empty parameter dictionary"""
        params = {}
        for param_type in ["kcat", "km", "kcat_km"]:
            prefix = param_type.replace("/", "_")
            params[f"{prefix}_value"] = None
            params[f"{prefix}_unit"] = None
            params[f"{prefix}_substrate"] = None
            params[f"{prefix}_notes"] = None
        return params

    def process_csv_file(
        self,
        input_file: str,
        output_file: str = None,
        start_row: int = 0,
        max_papers: Optional[int] = None,
        rate_limit_delay: float = 1.0,
    ):
        """
        Process CSV file and extract parameters from papers

        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file (default: input_file with _extracted suffix)
            start_row: Row to start processing from (for resuming)
            max_papers: Maximum number of papers to process (None for all)
            rate_limit_delay: Delay between API calls (seconds)
        """
        # Load CSV
        logger.info(f"📂 Loading CSV file: {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"📊 Loaded {len(df)} rows")

        # Prepare output file
        if output_file is None:
            output_file = input_file.replace(".csv", "_extracted.csv")

        # Add new columns if they don't exist
        for param_type, columns in self.param_columns.items():
            for col in columns:
                if col not in df.columns:
                    df[col] = None

        if "extraction_status" not in df.columns:
            df["extraction_status"] = None
        if "paper_content_type" not in df.columns:
            df["paper_content_type"] = None

        # Determine rows to process - use random sampling for test mode
        if max_papers and max_papers <= 10:  # Test mode - use random sampling
            available_rows = list(range(len(df)))
            random.seed(42)  # Reproducible random selection
            selected_rows = random.sample(
                available_rows, min(max_papers, len(available_rows))
            )
            selected_rows.sort()  # Sort for easier tracking
            logger.info(f"🎲 Random test mode: selected rows {selected_rows}")
            rows_to_process = selected_rows
            total_to_process = len(selected_rows)
        else:
            # Normal sequential processing
            end_row = min(start_row + max_papers, len(df)) if max_papers else len(df)
            total_to_process = end_row - start_row
            rows_to_process = list(range(start_row, end_row))

        logger.info(f"🎯 Processing {total_to_process} papers")

        # Set rate limits based on model (from official documentation)
        model_limits = {
            "gemini-2.5-pro": {"rpm": 5, "tpm": 250000, "rpd": 100},
            "gemini-2.5-flash": {"rpm": 10, "tpm": 250000, "rpd": 250},
            "gemini-2.5-flash-preview": {"rpm": 10, "tpm": 250000, "rpd": 250},
            "gemini-2.5-flash-lite": {"rpm": 15, "tpm": 250000, "rpd": 1000},
        }

        limits = model_limits.get(self.model_name, {"rpm": 15, "tpm": 250000, "rpd": 1000})
        rpm = limits["rpm"]
        tpm = limits["tpm"]
        rpd = limits["rpd"]
        logger.info(f"⚡ Using {self.model_name} (Limits: {rpm} RPM, {tpm//1000}K TPM, {rpd} RPD)")

        # Calculate safe delay based on token consumption
        # Average paper size ~80K chars ≈ ~20K tokens
        avg_tokens_per_paper = 20000
        safe_delay = max(
            60 / rpm,  # RPM-based delay
            (avg_tokens_per_paper * 60) / tpm  # TPM-based delay
        )

        logger.info(f"🕐 Calculated safe delay: {safe_delay:.1f} seconds between requests")

        # Process papers
        processed_count = 0
        save_interval = 10

        for idx in rows_to_process:
            row = df.iloc[idx]
            processed_count += 1

            # Progress indicator
            progress_pct = (processed_count / total_to_process) * 100
            logger.info(f"\n{'='*50}")
            logger.info(
                f"📍 Processing paper {processed_count}/{total_to_process} ({progress_pct:.1f}%) - Row {idx}"
            )
            logger.info(
                f"🧬 Enzyme: {row.get('enzyme_name', 'Unknown')}, Mutation: {row.get('mutation_type', 'Unknown')}"
            )

            # Skip if already processed
            if (
                pd.notna(df.at[idx, "extraction_status"])
                and df.at[idx, "extraction_status"] == "completed"
            ):
                logger.info("✅ Already processed, skipping...")
                continue

            # Get PubMed URL
            pubmed_url = row.get("pubmed_link")
            if pd.isna(pubmed_url) or not pubmed_url:
                logger.warning("⚠️ No PubMed URL found")
                df.at[idx, "extraction_status"] = "no_url"
                continue

            try:
                # Fetch paper content
                logger.info(f"🔍 Fetching content from: {pubmed_url}")
                paper_content, content_type = self.fetch_paper_content(pubmed_url)
                df.at[idx, "paper_content_type"] = content_type

                if paper_content:
                    # Extract parameters with Gemini
                    logger.info(f"🤖 Extracting parameters with Gemini...")
                    enzyme_info = {
                        "enzyme_name": row.get("enzyme_name"),
                        "mutation_type": row.get("mutation_type"),
                        "activity_change": row.get("activity_change"),
                    }

                    parameters = self.extract_parameters_with_gemini(
                        paper_content, enzyme_info
                    )

                    # Update dataframe
                    for key, value in parameters.items():
                        df.at[idx, key] = value

                    df.at[idx, "extraction_status"] = "completed"

                    # Log extracted values
                    if parameters.get("kcat_value"):
                        logger.info(
                            f"✨ kcat: {parameters['kcat_value']} {parameters.get('kcat_unit', '')}"
                        )
                    else:
                        logger.info("✨ kcat: None")
                    if parameters.get("km_value"):
                        logger.info(
                            f"✨ Km: {parameters['km_value']} {parameters.get('km_unit', '')}"
                        )
                    else:
                        logger.info("✨ Km: None")
                    if parameters.get("kcat_km_value"):
                        logger.info(
                            f"✨ kcat/Km: {parameters['kcat_km_value']} {parameters.get('kcat_km_unit', '')}"
                        )
                    else:
                        logger.info("✨ kcat/Km: None")
                else:
                    df.at[idx, "extraction_status"] = "fetch_failed"
                    logger.warning("❌ Failed to fetch paper content")

            except Exception as e:
                logger.error(f"❌ Error processing paper: {e}")
                df.at[idx, "extraction_status"] = "error"

            # Save progress
            if processed_count % save_interval == 0:
                df.to_csv(output_file, index=False)
                logger.info(f"💾 Saved progress to {output_file}")

            # Rate limiting delay - use calculated safe delay or user override
            if processed_count < total_to_process:
                actual_delay = rate_limit_delay if rate_limit_delay > 1.0 else safe_delay
                logger.debug(f"⏱️ Waiting {actual_delay:.1f}s before next request...")
                time.sleep(actual_delay)

        # Final save
        df.to_csv(output_file, index=False)
        logger.info(f"\n{'='*50}")
        logger.info(f"✅ Processing complete! Results saved to {output_file}")
        logger.info(f"📊 Processed {processed_count} papers")

        # Summary statistics
        completed = len(df[df["extraction_status"] == "completed"])
        with_kcat = len(df[df["kcat_value"].notna()])
        with_km = len(df[df["km_value"].notna()])
        with_efficiency = len(df[df["kcat_km_value"].notna()])

        logger.info(f"\n📈 Extraction Summary:")
        logger.info(f"  - Completed: {completed}")
        logger.info(f"  - Papers with kcat: {with_kcat}")
        logger.info(f"  - Papers with Km: {with_km}")
        logger.info(f"  - Papers with kcat/Km: {with_efficiency}")


def main():
    """Main function for command-line usage"""
    import argparse

    # Make the data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Extract enzyme parameters from research papers using Gemini"
    )
    parser.add_argument(
        "--input",
        default="data/enzyme_mutations_batch_1.csv",
        help="Input CSV file (default: batch 1)",
    )
    parser.add_argument(
        "--output", help="Output CSV file (default: input_extracted.csv)"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash-lite",
        choices=[
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-preview",
            "gemini-2.5-pro",
        ],
        help="Gemini model to use (flash-lite=1000 RPD, flash=250 RPD, pro=best quality but 100 RPD)",
    )
    parser.add_argument("--start", type=int, default=0, help="Starting row (0-indexed)")
    parser.add_argument("--max", type=int, help="Maximum papers to process")
    parser.add_argument(
        "--test", action="store_true", help="Test mode (process 5 random papers)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)",
    )

    args = parser.parse_args()

    # Test mode
    if args.test:
        args.max = 10
        logger.info("🧪 Running in test mode (10 random papers)")

    # Initialize extractor
    extractor = EnzymeParameterExtractor(model_name=args.model)

    # Process CSV
    extractor.process_csv_file(
        input_file=args.input,
        output_file=args.output,
        start_row=args.start,
        max_papers=args.max,
        rate_limit_delay=args.delay,
    )


if __name__ == "__main__":
    main()
