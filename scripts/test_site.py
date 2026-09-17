import tempfile
import unittest
from pathlib import Path
from prepare_site import output_path, resolve, sources, transform
from check_site import check


class ReadingSiteTests(unittest.TestCase):
    def setUp(self):
        self.catalog = {'Mathematics-Universe': [Path('03-高等数学/极限.md'), Path('README.md')], 'math-to-deep-learning': [Path('PART-02/反向传播.md')]}

    def test_wiki_links_resolve_locally_and_across_books(self):
        actual = transform('[[极限]] [[math-to-deep-learning/PART-02/反向传播|应用]]', Path('README.md'), 'Mathematics-Universe', self.catalog)
        self.assertIn('[极限](03-', actual)
        self.assertIn('https://caciniep.github.io/math-to-deep-learning/', actual)
        self.assertIn('[应用]', actual)

    def test_homepage_does_not_collide_with_global_index(self):
        self.assertNotEqual(str(output_path(Path('README.md'))).lower(), str(output_path(Path('INDEX.md'))).lower())

    def test_companion_markdown_links_are_verified(self):
        actual = transform('[应用](https://github.com/CacinieP/math-to-deep-learning/blob/main/PART-02/反向传播.md)', Path('README.md'), 'Mathematics-Universe', self.catalog)
        self.assertIn('https://caciniep.github.io/math-to-deep-learning/', actual)
        with self.assertRaises(ValueError):
            transform('[应用](https://github.com/CacinieP/math-to-deep-learning/blob/main/PART-02/不存在.md)', Path('README.md'), 'Mathematics-Universe', self.catalog)

    def test_missing_and_ambiguous_links_fail(self):
        with self.assertRaises(ValueError): resolve('不存在', self.catalog['Mathematics-Universe'])
        with self.assertRaises(ValueError): resolve('同名', [Path('a/同名.md'), Path('b/同名.md')])

    def test_literal_code_is_not_rewritten(self):
        source = '`[[not a link]]`\n```python\nx = "[[not a link]]"\n```\n'
        self.assertEqual(source, transform(source, Path('README.md'), 'Mathematics-Universe', self.catalog))

    def test_table_math_bars_and_display_math(self):
        actual = transform('| $P(A|B)$ |\n$$x^2$$\n', Path('README.md'), 'Mathematics-Universe', self.catalog)
        self.assertIn(r'$P(A\vert B)$', actual)
        import markdown
        rendered = markdown.markdown(actual, extensions=['tables', 'pymdownx.arithmatex'], extension_configs={'pymdownx.arithmatex': {'generic': True}})
        self.assertIn(r'P(A\vert B)', rendered)
        self.assertNotIn(r'P(A\|B)', rendered)
        self.assertIn('$$\nx^2\n$$', actual)

    def test_readme_and_directory_links(self):
        actual = transform('[首页](../README.md) [章节](./chapter/)', Path('a/file.md'), 'Mathematics-Universe', self.catalog)
        self.assertIn('(../index.md)', actual)
        self.assertIn('(./chapter/index.md)', actual)

    def test_sources_do_not_publish_private_or_generated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ['README.md', '.repo_memory/private.md', 'node_modules/pkg/README.md', '.site-docs/generated.md', 'chapter/lesson.md']:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# test')
            self.assertEqual(sources(root), [Path('README.md'), Path('chapter/lesson.md')])

    def test_broken_fragments_fail_the_site_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'index.html').write_text('<a href="chapter/#missing">Chapter</a>')
            (root / 'chapter').mkdir()
            (root / 'chapter/index.html').write_text('<h1 id="existing">Chapter</h1>')
            with self.assertRaises(SystemExit): check(root)
            (root / 'index.html').write_text('<a href="chapter/#existing">Chapter</a>')
            check(root)


if __name__ == '__main__': unittest.main()
