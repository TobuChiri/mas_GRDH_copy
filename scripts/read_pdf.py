"""Extract text from CGIS paper PDF."""
import PyPDF2, sys

with open('E:/code/mas_GRDH_copy/scripts/cgis_paper.pdf', 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    print(f'Total pages: {len(reader.pages)}')
    if reader.is_encrypted:
        reader.decrypt('')

    for i in range(len(reader.pages)):
        page = reader.pages[i]
        text = page.extract_text()
        print(f'\n{"="*60}')
        print(f'PAGE {i+1} (len={len(text)})')
        print(f'{"="*60}')
        if text:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            print(text)
        else:
            print('[EMPTY PAGE]')
