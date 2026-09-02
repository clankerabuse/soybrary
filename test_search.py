import os
import tempfile
import unittest

os.environ.setdefault("SOYBRARY_DATA_DIR", tempfile.mkdtemp(prefix="soybrary-test-"))

import library  # noqa: E402
import search  # noqa: E402

POSTS = [
    # id, tags, variant, subvariant, uploader
    (1, "gapejak smug glasses", "cobson", None, "anon1"),
    (2, "gapejak angry", "chudjak", "hdr", "anon2"),
    (3, "pointing text", "cobson", "wide", "soyteen"),
    (4, "big_boy stubble", None, None, "anon1"),
    (5, "coal", "markiplier", None, "poster3"),
]


def seed(db):
    for pid, tags, variant, subvariant, uploader in POSTS:
        db.save_post(pid, "completed", tags=tags, variant=variant,
                     subvariant=subvariant, uploader=uploader,
                     mime_type="image/png", extension="png")
    db.save_post(99, "failed", tags="gapejak")


class SearchTestCase(unittest.TestCase):
    def setUp(self):
        self.db = library.Database(":memory:")
        seed(self.db)
        self.conn = self.db.conn
        search.invalidate_counts()

    def tearDown(self):
        self.db.close()

    def ids(self, q, use_like=False):
        plan = search.build_like_query(q) if use_like else search.build_query(q, self.conn)
        rows = self.conn.execute(plan.select_sql, plan.params).fetchall()
        total = self.conn.execute(plan.count_sql, plan.params).fetchone()[0]
        self.assertEqual(total, len(rows))
        return {row[0] for row in rows}


class TestParseTerms(unittest.TestCase):
    def test_classifies_terms(self):
        self.assertEqual(
            search.parse_terms("gapejak variant:cobson subvariant:hdr 1234"),
            [("general", "gapejak"), ("variant", "cobson"),
             ("subvariant", "hdr"), ("id", "1234")])

    def test_bare_prefix_is_a_general_term(self):
        self.assertEqual(search.parse_terms("variant:"), [("general", "variant:")])

    def test_empty_query(self):
        self.assertEqual(search.parse_terms("   "), [])


class TestQueryResults(SearchTestCase):
    def test_empty_query_returns_completed_posts(self):
        self.assertEqual(self.ids(""), {1, 2, 3, 4, 5})

    def test_excludes_unfinished_posts(self):
        self.assertNotIn(99, self.ids("gapejak"))

    def test_general_tag(self):
        self.assertEqual(self.ids("gapejak"), {1, 2})

    def test_is_case_insensitive(self):
        self.assertEqual(self.ids("GaPeJaK"), self.ids("gapejak"))

    def test_terms_are_anded(self):
        self.assertEqual(self.ids("gapejak glasses"), {1})

    def test_variant_prefix(self):
        self.assertEqual(self.ids("variant:cobson"), {1, 3})

    def test_subvariant_prefix(self):
        self.assertEqual(self.ids("subvariant:hdr"), {2})

    def test_uploader(self):
        self.assertEqual(self.ids("anon1"), {1, 4})

    def test_id_term_matches_post_id(self):
        self.assertEqual(self.ids("3"), {3})

    def test_tag_prefix_matches(self):
        self.assertEqual(self.ids("gape"), {1, 2})

    def test_underscored_tag(self):
        self.assertEqual(self.ids("big_boy"), {4})

    def test_no_match(self):
        self.assertEqual(self.ids("nothinghere"), set())

    def test_newest_first(self):
        plan = search.build_query("", self.conn)
        rows = self.conn.execute(plan.select_sql, plan.params).fetchall()
        self.assertEqual([r[0] for r in rows], [5, 4, 3, 2, 1])

    def test_fts_results_ordered_newest_first(self):
        plan = search.build_query("gapejak", self.conn)
        rows = self.conn.execute(plan.select_sql, plan.params).fetchall()
        self.assertEqual([r[0] for r in rows], [2, 1])

    def test_matches_like_behaviour(self):
        for q in ("gapejak", "variant:cobson", "anon1", "gapejak glasses", "coal"):
            self.assertEqual(self.ids(q), self.ids(q, use_like=True), q)

    def test_never_returns_rows_the_scan_would_not(self):
        for q in ("gapejak", "3", "anon", "cob", "apejak", "big"):
            self.assertLessEqual(self.ids(q), self.ids(q, use_like=True), q)

    def test_fts_matches_are_a_subset_of_substring_matches(self):
        # "apejak" is mid-token: the index can't find it, the scan can.
        self.assertEqual(self.ids("apejak"), set())
        self.assertEqual(self.ids("apejak", use_like=True), {1, 2})


class TestQueryPlanning(SearchTestCase):
    def test_uses_index_for_ordinary_terms(self):
        self.assertTrue(search.build_query("gapejak", self.conn).used_fts)

    def test_index_scan_is_the_outer_loop(self):
        # If posts ends up on the outside the match is re-run per row, which
        # costs seconds instead of milliseconds on a real library.
        plan = search.build_query("gapejak", self.conn)
        steps = [r[-1] for r in self.conn.execute(
            "EXPLAIN QUERY PLAN " + plan.select_sql, plan.params)]
        self.assertTrue(steps[0].startswith("SCAN f VIRTUAL TABLE"), steps)
        self.assertTrue(any("PRIMARY KEY" in s for s in steps[1:]), steps)

    def test_count_also_drives_from_the_index(self):
        plan = search.build_query("gapejak", self.conn)
        steps = [r[-1] for r in self.conn.execute(
            "EXPLAIN QUERY PLAN " + plan.count_sql, plan.params)]
        self.assertTrue(steps[0].startswith("SCAN f VIRTUAL TABLE"), steps)

    def test_untokenisable_query_falls_back_to_scan(self):
        self.assertFalse(search.build_query("!!!", self.conn).used_fts)
        self.assertEqual(self.ids("!!!"), set())

    def test_quotes_do_not_break_the_query(self):
        for q in ('quote"inside', '""', 'a"b"c', "NEAR(", "*", "^", "tag*"):
            self.assertIsInstance(self.ids(q), set)

    def test_plain_query_needs_no_join(self):
        plan = search.build_query("", self.conn)
        self.assertNotIn("posts_fts", plan.select_sql)


class TestCountCache(unittest.TestCase):
    def setUp(self):
        search.invalidate_counts()

    def tearDown(self):
        search.invalidate_counts()

    def test_roundtrip(self):
        self.assertIsNone(search.cached_count("fts:x"))
        search.store_count("fts:x", 12)
        self.assertEqual(search.cached_count("fts:x"), 12)

    def test_invalidate(self):
        search.store_count("fts:x", 12)
        search.invalidate_counts()
        self.assertIsNone(search.cached_count("fts:x"))

    def test_cache_is_bounded(self):
        for i in range(search.COUNT_CACHE_LIMIT + 5):
            search.store_count(f"fts:{i}", i)
        self.assertLessEqual(len(search._count_cache), search.COUNT_CACHE_LIMIT)

    def test_keys_differ_per_strategy(self):
        conn = library.Database(":memory:")
        try:
            fts = search.build_query("gapejak", conn.conn)
            like = search.build_like_query("gapejak")
            self.assertNotEqual(fts.cache_key, like.cache_key)
        finally:
            conn.close()


class TestTagIndex(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="soybrary-tags-")
        self.path = os.path.join(self.dir, "tags.db")
        self.db = library.Database(self.path)
        seed(self.db)
        self.index = search.TagIndex()
        self.index.build(self.path)

    def tearDown(self):
        self.db.close()

    def test_prefix_match(self):
        self.assertEqual(set(self.index.prefix_search("gape")), {"gapejak"})

    def test_ranked_by_frequency(self):
        results = self.index.prefix_search("c")
        self.assertEqual(results[0], "cobson")  # 2 posts, ahead of "coal"
        self.assertIn("coal", results)

    def test_variants_indexed_with_and_without_prefix(self):
        self.assertIn("variant:cobson", self.index.prefix_search("variant:"))
        self.assertIn("cobson", self.index.prefix_search("cob"))

    def test_subvariants_indexed(self):
        self.assertIn("subvariant:hdr", self.index.prefix_search("subvariant:h"))

    def test_limit_is_respected(self):
        self.assertLessEqual(len(self.index.prefix_search("", limit=3)), 3)

    def test_unknown_prefix(self):
        self.assertEqual(self.index.prefix_search("zzz"), [])

    def test_ignores_unfinished_posts(self):
        self.db.save_post(500, "failed", tags="neverindexed")
        self.index.build(self.path)
        self.assertEqual(self.index.prefix_search("neverindexed"), [])


if __name__ == "__main__":
    unittest.main()
