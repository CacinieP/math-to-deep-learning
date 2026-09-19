import unittest
from pathlib import Path
from prepare_wiki import page_name, rewrite_url, convert, protect_inline_math


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
        self.assertIn('<a name="section-首页"></a>', result)
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


if __name__ == '__main__': unittest.main()
