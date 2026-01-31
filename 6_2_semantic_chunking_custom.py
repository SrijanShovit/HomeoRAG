import re
from typing import List
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

console = Console()


# ==========================================================
# Visualization helper
# ==========================================================
class RichVisualizer:
    def sentences(self, sentences: List[str]):
        console.print(Rule("[bold green]Sentence Split"))
        for i, s in enumerate(sentences):
            console.print(f"[bold]{i}[/bold]: {s}")

    def buffered_sentences(self, buffered: List[str]):
        console.print(Rule("[bold green]Buffered Sentences"))
        for i, s in enumerate(buffered):
            console.print(
                Panel(s, title=f"Buffered[{i}]", expand=False)
            )

    def similarities(self, sims: np.ndarray, threshold: float):
        console.print(Rule("[bold green]Adjacent Similarities"))

        table = Table(show_lines=True)
        table.add_column("Index (i → i+1)")
        table.add_column("Similarity", justify="right")
        table.add_column("≤ Threshold?", justify="center")

        for i, sim in enumerate(sims):
            table.add_row(
                f"{i} → {i+1}",
                f"{sim:.4f}",
                "YES" if sim <= threshold else "NO",
            )

        console.print(table)

    def breakpoints(self, sims, breaks):
        console.print(Rule("[bold green]Breakpoint Decisions"))

        table = Table(show_lines=True)
        table.add_column("Index (i→i+1)")
        table.add_column("Similarity")
        table.add_column("Break?")
        table.add_column("Reason")

        for i, (sim, brk) in enumerate(zip(sims, breaks)):
            reason = "SEMANTIC DROP" if brk else "-"
            table.add_row(
                f"{i} → {i+1}",
                f"{sim:.4f}",
                "[red]YES[/red]" if brk else "NO",
                reason,
            )

        console.print(table)

    def chunks(self, chunks: List[str]):
        console.print(Rule("[bold green]Final Chunks"))
        for i, chunk in enumerate(chunks):
            console.print(
                Panel(chunk, title=f"Chunk {i}", expand=False)
            )


# ==========================================================
# Semantic Chunker (similarity only)
# ==========================================================
class SemanticChunker:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        percentile: float = 90,
        threshold_type: str = "percentile",  # "std", "percentile", or "max" (default)
        min_sentences: int = 3,
        min_sentences_per_chunk: int = 2,
        max_sentences_per_chunk: int = 10,
        std_k: float = 1.0,
        buffer_size: int = 1,
        epsilon: float = 1e-4,
    ):
        self.model = SentenceTransformer(model_name)
        self.percentile = np.clip(percentile, 60, 95)
        self.threshold_type = threshold_type
        self.min_sentences = min_sentences
        self.min_sentences_per_chunk = min_sentences_per_chunk
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.std_k = std_k
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.viz = RichVisualizer()

    # --------------------------------------------------
    def _split_sentences(self, text: str) -> List[str]:
        return [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip()
        ]

    def _buffer_sentences(self, sentences: List[str]) -> List[str]:
        out = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            out.append(" ".join(sentences[start:end]))
        return out

    def _embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

    def _adjacent_similarities(self, emb: np.ndarray) -> np.ndarray:
        sims = []
        for i in range(len(emb) - 1):
            sims.append(
                cosine_similarity(
                    emb[i].reshape(1, -1),
                    emb[i + 1].reshape(1, -1),
                )[0][0]
            )
        return np.array(sims)

    def _is_degenerate(self, sims: np.ndarray) -> bool:
        return np.std(sims) < self.epsilon

    def _compute_threshold(self, sims: np.ndarray) -> float:
        p_cut = np.percentile(sims, 100 - self.percentile)
        z_cut = np.mean(sims) - self.std_k * np.std(sims)

        print(p_cut,z_cut)

        if self.threshold_type == "std":
            return z_cut
        elif self.threshold_type == "percentile":
            return p_cut
        else:  # "max"
            return max(p_cut, z_cut)

    def _decide_breakpoints(
        self,
        sims: np.ndarray,
    ) -> tuple[list[bool], float]:

        breaks = [False] * len(sims)
        threshold = self._compute_threshold(sims)

        for i in range(len(sims)):
            if sims[i] <= threshold:
                breaks[i] = True

        return breaks, threshold

    def _assemble_chunks(self, sentences, breaks):
        chunks, current = [], []
        last_break = -999

        for i, s in enumerate(sentences):
            current.append(s)

            can_break = i - last_break >= self.min_sentences_per_chunk
            force_break = len(current) >= self.max_sentences_per_chunk

            if i < len(breaks) and breaks[i] and can_break:
                chunks.append(" ".join(current))
                current = []
                last_break = i
            elif force_break:
                chunks.append(" ".join(current))
                current = []
                last_break = i

        if current:
            chunks.append(" ".join(current))
        return chunks

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------
    def split(self, text: str, visualize: bool = False) -> List[str]:
        sentences = self._split_sentences(text)

        if visualize:
            self.viz.sentences(sentences)

        if len(sentences) < self.min_sentences:
            return [text]

        buffered = self._buffer_sentences(sentences)
        if visualize:
            self.viz.buffered_sentences(buffered)

        embeddings = self._embed(buffered)
        sims = self._adjacent_similarities(embeddings)

        if self._is_degenerate(sims):
            return [" ".join(sentences)]

        breaks, threshold = self._decide_breakpoints(sims)

        if visualize:
            self.viz.similarities(sims, threshold)
            self.viz.breakpoints(sims, breaks)

        chunks = self._assemble_chunks(sentences, breaks)

        if visualize:
            self.viz.chunks(chunks)

        return chunks


if __name__ == "__main__":
    text = """MEDUSA
Jelly-fish
Whole face puffed œdematous-eyes, nose, ears, lips.
Skin.––Numbness; burning, pricking heat. Vesicular eruption especially on face, arms, shoulders, and breasts. Nettlerash (Apis; Chloral; Dulc).
Female.––Marked action on lacteal glands. The secretion of milk was established after lack of it in all previous confinements.
Relationship.––Compare: Pyrarara, Physalia (urticaria); Urtica, Homar, Sep.

MEL CUM SALE
Honey with Salt
Prolapsus uteri and chronic metritis, especially when associated with subinvolution and inflammation of the cervix. The special symptom leading to its selection is a feeling of soreness across the hypogastrium from ileum to ileum.
Uterine displacements, and in the commencement of metritis Sensation as if bladder were too full. Pain from sacrum towards pubes. Pain as if in ureters.
Dose.––Third to sixth potency. Honey for itching of anus and worms.

METHYLENUM COERULEUM
Aniline Dye
Methylene Blue
A remedy for neuralgia, neurasthenia, malaria; typhoid, here it diminishes the tympanites, delirium, and fever; pus infection. Tendency to tremor, chorea and epilepsy. Nephritis (acute parenchymatous), scarlatinal nephritis. Urine acquires a green color. Bladder irritation from its use antidoted by a little nutmeg.
Surgical kidney with large amount of pus in urine. Gonorrhœal rheumatism and cystitis. Backache, sciatica. Later states of apoplexy (Gisevius).
Dose.––3x attenuation. A 2 per cent solution locally, in chronic otitis with foul smelling discharge.
A 1 per cent aqueous solution for ulcers and abscesses of cornea."""

    SemanticChunker(percentile=85).split(text, visualize=True)
