from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

doc = Document(r'd:\vinuni_datathon2026\vinuni_datathon2026\model\technical_doc.docx')

def table_to_md(table):
    lines = []
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    if not rows:
        return ""
    # Header
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

md_lines = []
for block in doc.element.body:
    tag = block.tag.split("}")[-1]
    if tag == "p":
        para = Paragraph(block, doc)
        text = para.text.strip()
        if not text:
            md_lines.append("")
            continue
        style = para.style.name.lower()
        if "heading 1" in style:
            md_lines.append(f"# {text}")
        elif "heading 2" in style:
            md_lines.append(f"## {text}")
        elif "heading 3" in style:
            md_lines.append(f"### {text}")
        elif "heading 4" in style:
            md_lines.append(f"#### {text}")
        elif "list" in style or para.paragraph_format.left_indent:
            md_lines.append(f"- {text}")
        else:
            md_lines.append(text)
    elif tag == "tbl":
        tbl = Table(block, doc)
        md_lines.append("")
        md_lines.append(table_to_md(tbl))
        md_lines.append("")

content = "\n".join(md_lines)
out_path = r'd:\vinuni_datathon2026\vinuni_datathon2026\model\technical_doc.md'
with open(out_path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Done. Lines: {len(md_lines)}")
print("Preview (first 50 lines):")
print("\n".join(md_lines[:50]))
