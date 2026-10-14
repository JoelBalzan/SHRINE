"""Build the SHRINE Error Explorer into docs/index.html.

Sources, all in this folder:
  shell.html  layout, text, equations (MathJax), CSS
  core.js     mock FRB, dedispersion, DCT, filters and the SHRINE error maths
              (also loadable in Node: makeCore = require('./core.js'))
  app.js      figures, cards, steppers and interaction

Run `python explorer/build.py` (from anywhere), commit, push. GitHub Pages serves
docs/index.html at https://joelbalzan.github.io/SHRINE/ about a minute later.
"""
from pathlib import Path

here = Path(__file__).resolve().parent
shell = (here / 'shell.html').read_text()
core = (here / 'core.js').read_text().replace('if (typeof module !== "undefined") module.exports = makeCore;', '')
app = (here / 'app.js').read_text()

body = shell + '\n<script>\n' + core + '\n</script>\n<script>\n' + app + '\n</script>\n'
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')
page = head + body.replace('<div class="wrap">', '</head>\n<body>\n<div class="wrap">', 1) + '</body>\n</html>\n'

out = here.parent / 'docs' / 'index.html'
out.parent.mkdir(exist_ok=True)
out.write_text(page)
print(f'wrote {out} ({len(page):,} bytes)')
