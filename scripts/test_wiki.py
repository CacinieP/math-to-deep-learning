import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.parse import quote, unquote, urlsplit
from prepare_wiki import anchor_name, catalog, page_name, rewrite_url, convert, protect_inline_math


class WikiExportTests(unittest.TestCase):
    def setUp(self):
        self.catalogs = {
            'Mathematics-Universe': {
                Path('index.md'): {'name': 'Home', 'headings': ['首页']},
                Path('03-高等数学/极限.md'): {'name': '03-高等数学--极限', 'headings': ['定义']},
            },
            'math-to-deep-learning': {
                Path('index.md'): {'name': 'Home', 'headings': ['首页']},
                Path('PART-02/反向传播.md'): {'name': 'PART-02--反向传播', 'headings': ['链式法则']},
            },
        }

    def test_local_and_cross_book_links_keep_custom_anchors(self):
        local = rewrite_url('03-高等数学/极限.md#定义', Path('index.md'), 'Mathematics-Universe', self.catalogs)
        self.assertIn('/wiki/03-', local)
        self.assertIn('#section-', local)
        peer = rewrite_url('https://caciniep.github.io/math-to-deep-learning/PART-02/反向传播/#链式法则', Path('index.md'), 'Mathematics-Universe', self.catalogs)
        self.assertIn('/math-to-deep-learning/wiki/PART-02', peer)
        self.assertIn('#section-', peer)

    def test_missing_pages_and_fragments_fail(self):
        for url in ('missing.md', '03-高等数学/极限.md#missing'):
            with self.assertRaises(ValueError):
                rewrite_url(url, Path('index.md'), 'Mathematics-Universe', self.catalogs)

    def test_math_and_code_survive_and_headings_receive_anchors(self):
        source = '# 首页\n\n$$\nx^2\n$$\n\n```python\n# literal\nx = "[no](missing.md)"\n```\n'
        result = convert(source, Path('index.md'), 'Mathematics-Universe', self.catalogs)
        self.assertIn(f'<a name="{anchor_name("首页")}"></a>', result)
        self.assertIn('```math\nx^2\n```', result)
        self.assertIn('x = "[no](missing.md)"', result)

    def test_inline_tex_escapes_and_table_norm_are_protected(self):
        import re
        source = r'$\|x\|_2 < 1$'
        self.assertEqual(re.sub(r'\$([^$]+)\$', protect_inline_math, source), r'$`\Vert{}x\Vert{}_2 < 1`$')

    def test_matrix_row_separator_survives_gfm(self):
        source = '# 首页\n\n$$\n' + r'\begin{pmatrix}' + '\n1&2' + '\\'*2 + '\n3&4\n' + r'\end{pmatrix}' + '\n$$\n'
        result = convert(source, Path('index.md'), 'Mathematics-Universe', self.catalogs)
        self.assertIn('1&2' + '\\'*2 + ' \n', result)

    def test_unique_home_index_and_directory_names(self):
        self.assertEqual(page_name(Path('index.md')), 'Home')
        self.assertEqual(page_name(Path('03-高等数学/index.md')), '03-高等数学')
        self.assertNotEqual(page_name(Path('concept-index.md')), 'Home')

    def test_unicode_headings_use_distinct_stable_ascii_anchors(self):
        headings = ['定义', '定义_1', '链式法则', 'café', 'x→y']
        anchors = [anchor_name(heading) for heading in headings]
        self.assertEqual(len(anchors), len(set(anchors)))
        for heading, anchor in zip(headings, anchors):
            self.assertRegex(anchor, r'^section-[0-9a-f]{20}$')
            self.assertEqual(anchor_name(heading), anchor)
            # The live Wiki percent-encodes custom names and adds this prefix.
            published_name = 'user-content-' + quote(anchor, safe='').lower()
            self.assertEqual(unquote(published_name.removeprefix('user-content-')), anchor)

    def test_encoded_heading_link_matches_inserted_ascii_anchor(self):
        source = '# 首页\n\n[定义](03-高等数学/极限.md#%E5%AE%9A%E4%B9%89)\n'
        result = convert(source, Path('index.md'), 'Mathematics-Universe', self.catalogs)
        target = convert('# 定义\n', Path('03-高等数学/极限.md'), 'Mathematics-Universe', self.catalogs)
        href = rewrite_url('03-高等数学/极限.md#定义', Path('index.md'), 'Mathematics-Universe', self.catalogs)
        self.assertIn(href, result)
        self.assertIn(f'<a name="{urlsplit(href).fragment}"></a>', target)

    def test_catalog_rejects_anchor_collisions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / '.site-docs').mkdir()
            (root / '.site-docs/index.md').write_text('# 首页\n## 定义\n')
            (root / 'site').mkdir()
            (root / 'site/index.html').write_text('<article><h1 id="首页">首页</h1><h2 id="定义">定义</h2></article>')
            self.assertEqual(len(catalog(root)[Path('index.md')]['headings']), 2)
            with patch('prepare_wiki.anchor_name', return_value='section-collision'):
                with self.assertRaisesRegex(ValueError, 'Wiki anchor collision'):
                    catalog(root)

    def test_literal_matrix_brackets_are_not_wiki_links(self):
        source = '# 首页\n\n矩阵 A=[[2,1],[-1,4]]；`[[literal code]]`；$[[x]]$\n\n```python\nA = [[2, 1], [-1, 4]]\n```\n'
        result = convert(source, Path('index.md'), 'Mathematics-Universe', self.catalogs)
        self.assertIn('A=`[[2,1],[-1,4]]`', result)
        self.assertIn('`[[literal code]]`', result)
        self.assertIn('$`[[x]]`$', result)
        self.assertIn('A = [[2, 1], [-1, 4]]', result)


if __name__ == '__main__': unittest.main()
