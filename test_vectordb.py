import unittest
import math
import main

class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.db = main.VectorDB(16)
        main.load_demo(self.db)

    def test_distance_metrics(self):
        v1 = [1.0, 0.0, 0.0, 0.0] + [0.0] * 12
        v2 = [1.0, 0.0, 0.0, 0.0] + [0.0] * 12
        v3 = [0.0, 1.0, 0.0, 0.0] + [0.0] * 12

        # Distance to self should be 0
        self.assertAlmostEqual(main.cosine(v1, v2), 0.0, places=5)
        self.assertAlmostEqual(main.euclidean(v1, v2), 0.0, places=5)
        self.assertAlmostEqual(main.manhattan(v1, v2), 0.0, places=5)

        # Cosine distance between orthogonal vectors is 1.0
        self.assertAlmostEqual(main.cosine(v1, v3), 1.0, places=5)

    def test_search_consistency(self):
        # Query near CS domain
        q = [0.90, 0.80, 0.70, 0.60] + [0.10] * 12
        
        res_hnsw = self.db.search(q, 3, "cosine", "hnsw")
        res_bf = self.db.search(q, 3, "cosine", "bruteforce")
        res_kdt = self.db.search(q, 3, "cosine", "kdtree")

        self.assertEqual(len(res_hnsw["results"]), 3)
        self.assertEqual(len(res_bf["results"]), 3)
        self.assertEqual(len(res_kdt["results"]), 3)

        # Top result for CS query should belong to 'cs' category
        self.assertEqual(res_bf["results"][0]["category"], "cs")
        self.assertEqual(res_hnsw["results"][0]["category"], "cs")

    def test_insert_and_delete(self):
        initial_size = self.db.size()
        new_emb = [0.5] * 16
        item_id = self.db.insert("Test Item", "test", new_emb, main.cosine)
        
        self.assertEqual(self.db.size(), initial_size + 1)
        
        ok = self.db.remove(item_id)
        self.assertTrue(ok)
        self.assertEqual(self.db.size(), initial_size)

    def test_hnsw_info(self):
        info = self.db.hnsw_info()
        self.assertIn("topLayer", info)
        self.assertIn("nodeCount", info)
        self.assertGreater(info["nodeCount"], 0)

if __name__ == "__main__":
    unittest.main()
