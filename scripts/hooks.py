"""Use stable Unicode fragments and order navigation as a book, then maintenance."""
from pathlib import Path
from pymdownx.slugs import slugify


def on_config(config):
    config.mdx_configs.setdefault('toc', {})['slugify'] = slugify(case='lower')
    root = Path(config.docs_dir)

    def chapter(folder):
        entries = []
        index = folder / 'index.md'
        if index.is_file():
            entries.append({'概览': index.relative_to(root).as_posix()})
        for child in sorted(folder.iterdir()):
            if child.is_dir():
                entries.append({child.name: chapter(child)})
            elif child.suffix == '.md' and child.name != 'index.md':
                entries.append(child.relative_to(root).as_posix())
        return entries

    nav = [{'开始阅读': 'index.md'}]
    if (root / 'concept-index.md').exists():
        nav.append({'全局索引': 'concept-index.md'})
    folders = [p for p in root.iterdir() if p.is_dir() and p.name not in {'assets', 'docs'}]
    for folder in sorted(folders, key=lambda p: (p.name.startswith('APPENDIX'), p.name)):
        nav.append({folder.name: chapter(folder)})
    maintenance = [{'贡献指南': 'CONTRIBUTING.md'}, {'更新记录': 'CHANGELOG.md'}]
    if (root / 'docs').is_dir():
        maintenance.append({'审核报告': chapter(root / 'docs')})
    nav.append({'项目维护': maintenance})
    config.nav = nav
    return config
