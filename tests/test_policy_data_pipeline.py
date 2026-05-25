import unittest

from policy_data_pipeline import (
    Chunk,
    SimpleHashEmbedder,
    build_chroma_payload,
    chunk_text,
    clean_policy_text,
)


class PolicyDataPipelineTests(unittest.TestCase):
    def test_clean_policy_text_removes_noise(self):
        raw = """
        QuePlan.cl
        POLIZA VIDA ACTIVA
        Página 1 de 4
        Índice........... 2

        Cobertura principal de salud.
        Cobertura secundaria de accidentes.

        POLIZA VIDA ACTIVA
        Página 2 de 4
        www.queplan.cl
        """

        cleaned = clean_policy_text(raw)

        self.assertNotIn("QuePlan.cl", cleaned)
        self.assertNotIn("Página 1", cleaned)
        self.assertNotIn("Índice", cleaned)
        self.assertNotIn("www.queplan.cl", cleaned)
        self.assertIn("Cobertura principal de salud.", cleaned)
        self.assertIn("Cobertura secundaria de accidentes.", cleaned)

    def test_chunk_text_splits_large_content(self):
        text = (
            "Cobertura A: atención ambulatoria y hospitalaria para el titular y cargas. "
            "Incluye urgencias, exámenes de laboratorio y seguimiento.\n\n"
            "Cobertura B: medicamentos con copago, cirugías programadas y prestaciones dentales. "
            "Aplica red preferente y tope anual por beneficiario.\n\n"
            "Exclusiones: enfermedades preexistentes no declaradas, tratamientos estéticos y actos ilícitos."
        )

        chunks = chunk_text(text, chunk_size=140, overlap=20)

        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 140)
            self.assertTrue(chunk.text)

    def test_build_chroma_payload_returns_expected_shapes(self):
        chunks = [
            Chunk(text="Cobertura hospitalaria y urgencias.", start_char=0, end_char=36),
            Chunk(text="Prestaciones dentales y medicamentos.", start_char=37, end_char=74),
        ]

        payload = build_chroma_payload(chunks, source_id="queplan-123", embedder=SimpleHashEmbedder(dimension=16))

        self.assertEqual(payload["ids"], ["queplan-123-chunk-0", "queplan-123-chunk-1"])
        self.assertEqual(len(payload["documents"]), 2)
        self.assertEqual(len(payload["metadatas"]), 2)
        self.assertEqual(len(payload["embeddings"]), 2)
        self.assertEqual(len(payload["embeddings"][0]), 16)
        self.assertEqual(payload["metadatas"][1]["chunk_index"], 1)


if __name__ == "__main__":
    unittest.main()
