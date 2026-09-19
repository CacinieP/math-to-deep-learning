"""Prepare a reproducible reading site from the repository's Markdown sources."""
from __future__ import annotations
import argparse
import posixpath
import re
import shutil
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REPOS = ('Mathematics-Universe', 'math-to-deep-learning')
EXCLUDED = {'.git', '.repo_memory', '.venv', 'node_modules', 'site', '.site-docs', 'scripts', 'web'}


def sources(root):
    return sorted(p.relative_to(root) for p in root.rglob('*.md')
                  if not any(part in EXCLUDED or part.startswith('.') for part in p.relative_to(root).parts))


def output_path(path):
    # Avoid README/index and INDEX collisions on case-insensitive filesystems.
    return {'README.md': Path('index.md'), 'INDEX.md': Path('concept-index.md')}.get(path.as_posix(), path)


def site_url(repo, path):
    path = output_path(path)
    route = path.with_suffix('').as_posix()
    if path.name == 'index.md':
        route = path.parent.as_posix()
    return f'https://caciniep.github.io/{repo}/' + (quote(route, safe='/') + '/' if route != '.' else '')


def resolve(target, paths):
    target = unquote(target).removesuffix('.md')
    matches = [p for p in paths if p.with_suffix('').as_posix() == target]
    if not matches and '/' not in target:
        matches = [p for p in paths if p.stem == target]
    if len(matches) != 1:
        raise ValueError(f'unresolved or ambiguous wiki link: {target!r}')
    return matches[0]


def transform(text, source, repo, catalog):
    fence = None
    result = []
    for line in text.splitlines(keepends=True):
        match = re.match(r'^\s*(`{3,}|~{3,})', line)
        if match:
            mark = match.group(1)
            if fence is None:
                fence = mark
            elif mark[0] == fence[0] and len(mark) >= len(fence):
                fence = None
            result.append(line)
            continue
        if fence:
            result.append(line)
            continue
        def wiki(match):
            value = match.group(1).replace('\\|', '|')
            dest, sep, label = value.partition('|')
            target, hashmark, anchor = dest.partition('#')
            target_repo = repo
            if target.split('/')[0] in REPOS:
                target_repo, target = target.split('/', 1)
            path = resolve(target, catalog[target_repo])
            url = site_url(target_repo, path) if target_repo != repo else quote(posixpath.relpath(output_path(path).as_posix(), output_path(source).parent.as_posix()), safe='/')
            if hashmark:
                url += '#' + anchor
            return f'[{label if sep else Path(target).stem}]({url})'
        # Inline code is literal, including examples of wiki syntax.
        chunks = re.split(r'(`+[^`]*`+)', line)
        for i in range(0, len(chunks), 2):
            chunks[i] = re.sub(r'\[\[([^\]]+)\]\]', wiki, chunks[i])
            # Markdown tables split raw TeX bars before math rendering.
            if line.lstrip().startswith('|'):
                chunks[i] = re.sub(r'(?<!\\)\$[^$\n]+(?<!\\)\$', lambda m: re.sub(r'(?<!\\)\|', r'\\vert{}', m[0]), chunks[i])
        line = ''.join(chunks)
        # README becomes the homepage; directory links gain generated indexes.
        def local_link(match):
            prefix, url, suffix = match.groups()
            parsed = urlsplit(url)
            if parsed.netloc == 'github.com':
                parts = unquote(parsed.path).strip('/').split('/')
                if len(parts) >= 5 and parts[0] == 'CacinieP' and parts[1] in REPOS and parts[2:4] == ['blob', 'main']:
                    peer_repo = parts[1]
                    peer_path = Path('/'.join(parts[4:]))
                    if peer_path.suffix == '.md':
                        if peer_path not in catalog[peer_repo]:
                            raise ValueError(f'missing companion page: {url}')
                        return prefix + site_url(peer_repo, peer_path) + (('#' + parsed.fragment) if parsed.fragment else '') + suffix
            if parsed.scheme or parsed.netloc or not parsed.path:
                return match[0]
            path = unquote(parsed.path)
            if path.endswith('INDEX.md'):
                path = path[:-len('INDEX.md')] + 'concept-index.md'
            if path.endswith('README.md'):
                path = path[:-len('README.md')] + 'index.md'
            if path.endswith('/'):
                path += 'index.md'
            return prefix + quote(path, safe='/.-') + (('#' + parsed.fragment) if parsed.fragment else '') + suffix
        chunks = re.split(r'(`+[^`]*`+)', line)
        for i in range(0, len(chunks), 2):
            chunks[i] = re.sub(r'(\]\()([^\s)]+)(\))', local_link, chunks[i])
        line = ''.join(chunks)
        # Accept GitHub's single-line display math as a standalone block.
        line = re.sub(r'^\s*\$\$(.+)\$\$\s*$', lambda m: '\n$$\n' + m[1] + '\n$$\n\n', line)
        result.append(line)
    if fence:
        raise ValueError(f'unclosed fenced block in {source}')
    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--peer', type=Path, required=True, help='Checkout of the companion repository')
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load((ROOT / 'mkdocs.yml').read_text())
    repo = config['repo_name'].split('/')[-1]
    peer_name = next(name for name in REPOS if name != repo)
    if not (args.peer / 'README.md').is_file():
        raise SystemExit('Companion repository checkout is missing')
    catalog = {repo: sources(ROOT), peer_name: sources(args.peer)}
    destination = ROOT / '.site-docs'
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir()
    errors = []
    for source in catalog[repo]:
        try:
            content = transform((ROOT / source).read_text(), source, repo, catalog)
        except ValueError as error:
            errors.append(f'{source}: {error}')
            continue
        target = destination / output_path(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    if errors:
        raise SystemExit('\n'.join(errors))
    # Every linked chapter directory gets a real landing page.
    directories = sorted({parent for path in catalog[repo] for parent in path.parents if parent != Path('.')}, key=lambda p: (-len(p.parts), p.as_posix()))
    for directory in directories:
        folder = destination / directory
        if (folder / 'index.md').exists():
            continue
        entries = []
        for child in sorted(folder.iterdir()):
            if child.name == 'index.md':
                continue
            if child.is_dir():
                entries.append(f'- [{child.name}]({quote(child.name)}/index.md)')
            elif child.suffix == '.md':
                entries.append(f'- [{child.stem}]({quote(child.name)})')
        (folder / 'index.md').write_text('# ' + directory.name + '\n\n' + '\n'.join(entries) + '\n')
    shutil.copy2(ROOT / 'LICENSE', destination / 'LICENSE')
    shutil.copytree(ROOT / 'web', destination / 'assets')
    mathjax = ROOT / 'node_modules/mathjax/es5'
    if not mathjax.is_dir():
        raise SystemExit('Run npm ci before preparing the site')
    shutil.copytree(mathjax, destination / 'assets/mathjax')
    # Published manifest enables validation of cross-repository routes and anchors.
    print(f'Prepared {len(catalog[repo])} source pages for {repo}')


if __name__ == '__main__':
    main()
