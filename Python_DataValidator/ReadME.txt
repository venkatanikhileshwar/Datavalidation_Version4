Data Validation App

A Streamlit-based Data Validation Tool to compare data between files (Excel/CSV/TXT/PDF) and databases (SQLite or others).
Supports checks like duplicates, null issues, type checks, value mismatches, missing records, and generates a detailed Excel report with charts.

🚀 Installation

Clone or copy the repo into your system.

Create & activate a Python virtual environment (recommended):

# Windows
python -m venv .venv
.venv\Scripts\activate

# Mac/Linux
python3 -m venv .venv
source .venv/bin/activate


Install required dependencies:

pip install -r requirements.txt


Initialize the sample database (if not already present):

python create_db.py


This will create a test_data.sqlite file with sample tables.

▶️ Running the App

Launch Streamlit:

streamlit run app.py


Open the browser link provided (usually http://localhost:8501
).

📂 Project Structure & File Importance
├── app.py                  # Main Streamlit app (UI + validation workflow)
├── create_db.py            # Script to generate a sample SQLite DB for testing
├── test_data.sqlite        # Example SQLite database file
├── requirements.txt        # Python dependencies
├── validation_report.xlsx  # Auto-generated report after validation
├── config/
│   └── appconfig.json      # Database connection config + defaults
├── utils/
│   ├── io_utils.py         # File input/output utilities (detect file type, read Excel/CSV/TXT/PDF)
│   ├── sql_utils.py        # SQL handling (connection, safe query validation, preview execution)
│   ├── mapping_utils.py    # Mapping logic (file headers ↔ DB columns, ignore handling)
│   ├── validate_utils.py   # Core validation checks (duplicates, nulls, types, mismatches)
│   ├── report_utils.py     # Report writer (Excel export, conditional formatting, donut chart)
│   └── __pycache__/        # Auto-generated Python cache files

🛠 Features

Upload Excel, CSV, TXT, PDF.

Run custom SQL SELECT queries (supports JOIN, GROUP BY, ORDER BY, etc.).

Map file headers ↔ database columns via dropdowns.

Validations:

Duplicates (flat + grouped views)

Null checks

Type checks (number/date inference)

Value mismatches

Missing in DB / Missing in File

Detailed Excel report:

Separate sheets for each validation

Merged Duplicates Resolve sheet

Conditional formatting for mismatches

Donut chart summarizing issues

Tracks validation time in UI & report.

📊 Example Output

Streamlit UI → Expander tables + KPIs + checklist

Excel Report → Tabs for Summary, Duplicates, Null Issues, Type Issues, Mismatches, Missing Data.