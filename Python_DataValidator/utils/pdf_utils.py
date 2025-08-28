import os, tempfile
import pandas as pd
import pdfplumber

def convert_uploaded_pdf_to_csv_temp(uploaded_file) -> str | None:
    tmpdir = tempfile.mkdtemp(prefix="pdf2csv_")
    pdf_path = os.path.join(tmpdir, "input.pdf")
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    column_names = None
    column_names_lower = None
    all_rows = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_x_tolerance": 5,
                "intersection_y_tolerance": 5,
            }) or []

            for table in tables:
                if table and len(table) > 1:
                    header = table[0]
                    rows = table[1:]
                    normalized_header = [(col.strip().lower() if col else "") for col in header]

                    if column_names is None:
                        column_names = [(col.strip() if col else "") for col in header]
                        column_names_lower = [(col.lower() if col else "") for col in column_names]

                    if normalized_header == column_names_lower:
                        data_rows = rows
                    else:
                        data_rows = table

                    for row in data_rows:
                        while len(row) < len(column_names):
                            row.append("")
                        all_rows.append(row)

    if not all_rows or not column_names:
        return None

    df = pd.DataFrame(all_rows, columns=column_names)
    csv_path = os.path.join(tmpdir, "output.csv")
    df.to_csv(csv_path, index=False)
    return csv_path
