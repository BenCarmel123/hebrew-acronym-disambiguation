"""Check the compiled draft without touching research data or running models."""
from pathlib import Path
import re
import subprocess

log = Path('build/main.log').read_text()
aux = Path('build/main.aux').read_text()
problems = []
for marker in ('undefined citations', 'undefined references', 'Missing character:', 'Overfull \\hbox', 'Overfull \\vbox'):
    if marker in log:
        problems.append(marker)
match = re.search(r'\\newlabel\{bodyend\}\{\{[^}]*\}\{(\d+)\}', aux)
if not match:
    problems.append('Missing body-end page label')
else:
    # Main-content floats may land after the body-end label. Count their pages too.
    float_pages = [int(page) for page in re.findall(
        r'\\newlabel\{(?:tab|fig):[^}]+\}\{\{[^}]*\}\{(\d+)\}', aux)]
    body_page = max([int(match.group(1)), *float_pages])
    print(f'Main content, including floats, ends on page {body_page}; course maximum is 8.')
    if body_page > 8:
        problems.append('Body exceeds eight pages')
print(subprocess.check_output(['pdfinfo', 'build/main.pdf'], text=True).split('Pages:')[1].splitlines()[0].strip() + ' total PDF pages')
if problems:
    raise SystemExit('Build checks failed: ' + '; '.join(problems))
print('References, glyphs, boxes and page limit passed. Visual review is still required.')
