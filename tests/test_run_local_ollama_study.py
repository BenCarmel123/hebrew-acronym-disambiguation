"""Check the local Ollama runner's system description without a server."""
import unittest

from hebrew_acronyms import run_local_ollama_study as local


class LocalOllamaStudyTests(unittest.TestCase):
    def test_dictalm_uses_seed_42_greedy_options_and_the_inspected_digest(self):
        system = local.build_system("dictalm", "hf.co/example/dictalm:Q4_K_M", {"digest": "abc"})
        self.assertEqual(system["provider"], "qwen")
        self.assertEqual(system["settings"]["expected_digest"], "abc")
        self.assertEqual(system["settings"]["options"], {"temperature": 0, "seed": 42, "num_predict": 512})

    def test_only_ollama_systems_are_accepted(self):
        with self.assertRaises(ValueError):
            local.build_system("xai", "grok-4.7", {"digest": "abc"})


if __name__ == "__main__":
    unittest.main()
