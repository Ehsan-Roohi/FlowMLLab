"""Build GitHub weekly landing pages from the maintained course indexes/gallery."""
from pathlib import Path
import re
import subprocess
import posixpath
from urllib.parse import urlsplit, unquote

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    p = ROOT / path
    if p.exists():
        return p.read_text()
    return subprocess.check_output(['git', 'show', f'HEAD:{path}'], cwd=ROOT, text=True)

def slug(week):
    parts = week.split('.')
    return 'week' + parts[0].zfill(2) + (('_' + parts[1]) if len(parts) > 1 else '')

def links(text):
    return re.findall(r'\[([^\]]+)\]\(([^)]+)\)', text)

def rebase(text, prefix='../../'):
    def replace(m):
        url = m[1]
        if url.startswith(('http:', 'https:', '#', 'mailto:', '/')):
            return m[0]
        return '](' + prefix + url + ')'
    return re.sub(r'\]\(([^)]+)\)', replace, text)

def main():
    home = read('README.md')
    launch = read('notebooks/README.md')
    course = read('COURSE_MAP.md')
    table = home.split('## Continue through the course', 1)[1].split('## Results gallery', 1)[0]
    entries = []
    for line in table.splitlines():
        m = re.match(r'\| \[([\d.]+)\]\([^)]+\) \| (.+?) \| (.+?) \| (.+?) \|$', line)
        if m:
            entries.append(dict(week=m[1], topic=m[2], lablinks=m[3], lecture=m[4]))
    assert len(entries) >= 16
    headings = list(re.finditer(r'^### Week ([\d.]+) [—–-]+ (.+)$', home, re.M))
    gallery = {}
    for i, m in enumerate(headings):
        end = headings[i+1].start() if i+1 < len(headings) else home.index('## Reuse and contribute')
        body = home[m.end():end].strip()
        body = re.sub(r'^\*\*\[Open Week .*?\n\n', '', body, flags=re.M)
        gallery[m[1]] = (m[2], body)
    assert set(gallery) == {e['week'] for e in entries}

    labs = {}
    for line in launch.splitlines():
        if not line.startswith('| Week'):
            continue
        cells = [x.strip() for x in line.strip('|').split('|')]
        module, label, launches = cells[:3]
        if module == 'Weeks 5 to 6':
            weeks = ['5', '6']
        else:
            match = re.match(r'Week ([\d.]+)', module)
            if not match:
                continue
            weeks = [match[1]]
        for linklabel, url in links(launches):
            if '.ipynb' not in url:
                continue
            colab = url.startswith('https://colab.research.google.com/')
            path = url.split('/blob/main/', 1)[1] if colab else posixpath.normpath('notebooks/' + url)
            title = label if 'historical' not in linklabel.lower() else 'Historical geometry-operator audit (optional)'
            for week in weeks:
                labs.setdefault(week, []).append((title, path, colab))
    # Conceptual goals come from the already-maintained course map.
    goals = {}
    for line in course.splitlines():
        m = re.match(r'\| \[([\d.]+)(?:[ABC]| Lab 3)?\]\([^)]+\) \| (.+?) \|', line)
        if m:
            goals.setdefault(m[1], []).append(m[2])

    for idx, entry in enumerate(entries):
        week = entry['week']; folder = slug(week)
        title, body = gallery[week]
        image = re.search(r'!\[[^\]]*\]\([^)]+\)', body)
        assert image, f'No image: {week}'
        hero = image[0]
        body = body[:image.start()] + body[image.end():]
        previous = entries[idx-1]['week'] if idx else None
        following = entries[idx+1]['week'] if idx+1 < len(entries) else None
        navigation = '[Course home](../../README.md) · [All weeks](../README.md)'
        if previous:
            navigation += f' · [← Week {previous}](../{slug(previous)}/README.md)'
        if following:
            navigation += f' · [Week {following} →](../{slug(following)}/README.md)'
        parts = [f'# Week {week} — {title}', navigation, entry['topic'] + '.', rebase(hero),
                 '## Lecture and notebooks', '**Lecture:** ' + rebase(entry['lecture']),
                 '| Notebook | Read | Run / setup |\n| --- | --- | --- |']
        for label, path, colab in labs.get(week, []):
            preview = f'[Open notebook](../../{path})'
            setup = posixpath.dirname(path) + '/README.md'
            if colab:
                run = f'[Open in Colab](https://colab.research.google.com/github/Ehsan-Roohi/FlowMLLab/blob/main/{path})'
            else:
                run = f'[Setup and reproduction guide](../../{setup})'
            parts[-1] += f'\n| {label} | {preview} | {run} |'
        assert labs.get(week), f'No notebooks: {week}'
        setup_links = [(label,url) for label,url in links(entry['lablinks']) if '.ipynb' not in url]
        if setup_links:
            parts.append('**Module guides:** ' + ' · '.join(rebase(f'[{label}]({url})') for label,url in setup_links))
        if week in goals:
            parts.extend(['## What you will work on', '\n'.join('- ' + rebase(g) for g in dict.fromkeys(goals[week]))])
        if week in ['5', '6']:
            sequence = ('Start with P0, choose one project track, and freeze its question, baseline and evaluation plan.' if week == '5'
                        else 'Continue the track selected in Week 5; finish its physical checks, reproducible results and report.')
            sequence += ' The lecture and project notebooks are shared across Weeks 5–6; you do not need to complete every track. P6 requires CUDA. Take the sparse-sensing companion after Week 7.'
        else:
            sequence = 'Read the lecture, then work through the notebooks in the listed order. Read each notebook’s setup instructions before running its cells; use the figures and evidence below to interpret the results.'
        if week == '14':
            sequence += ' This module uses a separate local environment; hosted Run All is not verified. Follow its setup guide.'
        if week == '15':
            sequence += ' Use the complete post-audit notebook with a local clone and its linked evidence archive. The historical notebook is retained for comparison.'
        if week == '16':
            sequence += ' Follow the assignment guide to distinguish retained CFD evidence from computations reproduced in the notebook.'
        if week == '4.1':
            sequence += ' The Week 4 lecture is the shared companion; this notebook contains the additional ROM theory.'
        parts.extend(['## Suggested route', sequence, '## Experiment and results', rebase(body.strip()), '---', navigation])
        dest = ROOT / 'weeks' / folder / 'README.md'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('\n\n'.join(parts) + '\n')

    # The main table now opens landing pages, while direct lecture/lab shortcuts stay available.
    for entry in entries:
        week = entry['week']; target = f"weeks/{slug(week)}/README.md"
        home = re.sub(r'(?m)^(\| \[' + re.escape(week) + r'\]\()[^)]+(\) \|)',lambda m:m[1]+target+m[2],home)
        pattern = r'(?m)^(### Week '+re.escape(week)+r' [—–-]+ .+\n)'
        home = re.sub(pattern,lambda m:m[1]+f'\n**[Open Week {week}: lecture, notebooks and guide]({target})**\n',home)
        # Idempotence: remove adjacent repeated links from a previous build.
        link = f'**[Open Week {week}: lecture, notebooks and guide]({target})**'
        home = re.sub(re.escape(link)+r'\s*'+re.escape(link),lambda m:link,home)
    home = home.replace('Each week has its own row, including the incremental laboratories.',
                        'Open any week number below for its own page with the topic image, lecture, notebooks and learning guide. Incremental laboratories also have dedicated pages.')
    home = home.replace('[Course map](COURSE_MAP.md) · [All notebooks]', '[Weekly pages](weeks/README.md) · [Course map](COURSE_MAP.md) · [All notebooks]')
    # Avoid duplicate weekly-page entry on repeated builds.
    home = home.replace('[Weekly pages](weeks/README.md) · [Weekly pages](weeks/README.md)', '[Weekly pages](weeks/README.md)')
    (ROOT/'README.md').write_text(home)
    index = ['# Explore FlowMLLab by week', '[← Course home](../README.md)',
             'Each page brings the topic image, lecture, notebooks and learning guide together. Weeks 5 and 6 have separate pages and share a project pack.',
             '| Week | Topic |\n| --- | --- |']
    for entry in entries:
        title = gallery[entry['week']][0]
        index[-1] += f"\n| [Week {entry['week']}]({slug(entry['week'])}/README.md) | {title} |"
    (ROOT/'weeks/README.md').write_text('\n\n'.join(index)+'\n')
    for path in ['notebooks/README.md', 'lectures/README.md', 'COURSE_MAP.md']:
        text = read(path)
        prefix = '' if path == 'COURSE_MAP.md' else '../'
        note = f'**Browse by week:** [Open the weekly pages]({prefix}weeks/README.md) for each topic’s image, lecture, notebooks and guide.'
        if note not in text:
            head, rest = text.split('\n', 1)
            text = head + '\n\n' + note + '\n' + rest
        dest = ROOT/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(text)
    print(f'Built {len(entries)} weekly pages with {sum(len(v) for v in labs.values())} notebook links (shared project tracks appear in Weeks 5 and 6).')

if __name__ == '__main__':
    main()
