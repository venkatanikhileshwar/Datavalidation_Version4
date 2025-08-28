import pandas as pd

def export_excel(path: str, results: dict) -> None:
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        wb = writer.book

        # 1) Write everything except Summary first
        for name, obj in results.items():
            if name == "Summary":
                continue
            if isinstance(obj, pd.DataFrame) and not obj.empty:
                sheet = name[:31]
                obj.to_excel(writer, sheet_name=sheet, index=False)
                ws = writer.sheets[sheet]

                # Autofilter + widths
                if len(obj.columns) > 0:
                    ws.autofilter(0, 0, len(obj), len(obj.columns) - 1)
                for j, col in enumerate(obj.columns):
                    try:
                        width = min(max(10, int(obj[col].astype(str).str.len().max()) + 2), 60)
                    except Exception:
                        width = 18
                    ws.set_column(j, j, width)

                # Highlight mismatch flags
                if sheet == "Value_Mismatches":
                    fmt_bad = wb.add_format({"bg_color": "#FFC7CE"})
                    for j, col in enumerate(obj.columns):
                        if str(col).endswith("__mismatch"):
                            ws.conditional_format(
                                1, j, len(obj), j,
                                {"type": "cell", "criteria": "==", "value": True, "format": fmt_bad}
                            )

        # 2) Summary LAST (no donut chart to keep it simple)
        summary_df = results.get("Summary")
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            sheet = "Summary"
            summary_df.to_excel(writer, sheet_name=sheet, index=False)
            ws = writer.sheets[sheet]

            if len(summary_df.columns) > 0:
                ws.autofilter(0, 0, len(summary_df), len(summary_df.columns) - 1)
            for j, col in enumerate(summary_df.columns):
                try:
                    width = min(max(10, int(summary_df[col].astype(str).str.len().max()) + 2), 60)
                except Exception:
                    width = 18
                ws.set_column(j, j, width)
