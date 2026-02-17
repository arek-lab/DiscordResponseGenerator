import gradio as gr
import asyncio
import glob
import os
import sys
import queue
import threading

from not_used.message_filter import process_messages
from utils.process_graphs import process_candidates_with_batching

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def find_new_json_files(before: dict[str, float]) -> list[str]:
    after = {f: os.path.getmtime(f) for f in glob.glob(os.path.join(BASE_DIR, "*.json"))}
    return sorted(f for f, mtime in after.items() if f not in before or mtime > before[f])


class QueueWriter:
    """Zastępuje sys.stdout – każda linia trafia do kolejki w czasie rzeczywistym."""
    def __init__(self, q: queue.Queue):
        self.q = q
        self.buf = ""

    def write(self, text: str):
        self.buf += text
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.q.put(line)

    def flush(self):
        if self.buf:
            self.q.put(self.buf)
            self.buf = ""


def run_pipeline_thread(uploaded_file, q: queue.Queue):
    """Uruchamia pipeline w osobnym wątku; wyniki trafiają do kolejki q."""

    async def _run():
        print("🚀 Start przetwarzania...")

        with open(uploaded_file, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"📄 Wczytano plik ({len(text):,} znaków).")

        candidates, all_messages = process_messages(text)
        print(f"🔍 Znaleziono kandydatów: {len(candidates)}")
        print("⏳ Przetwarzanie wsadowe – proszę czekać…\n")

        snapshot_before = {f: os.path.getmtime(f) for f in glob.glob(os.path.join(BASE_DIR, "*.json"))}

        prev_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        results = await process_candidates_with_batching(candidates, max_concurrent=15, batch_size=20)
        os.chdir(prev_cwd)

        successful = [r for r in results if r["status"] == "success"]
        failed     = [r for r in results if r["status"] == "error"]

        print("\n" + "=" * 60)
        print("📈 Podsumowanie")
        print("=" * 60)
        print(f"   ✅ Sukces : {len(successful)}")
        print(f"   ❌ Błędy  : {len(failed)}")

        new_files = find_new_json_files(snapshot_before)
        if new_files:
            for f in new_files:
                print(f"   📁 {os.path.basename(f)} — gotowy do pobrania ⬇️")
            # Przekaż listę plików przez kolejkę jako specjalny znacznik
            q.put(("__files__", new_files))
        else:
            print(f"   ⚠️  Brak nowych plików JSON w: {BASE_DIR}")

    old_stdout = sys.stdout
    sys.stdout = QueueWriter(q)
    try:
        asyncio.run(_run())
    except Exception as exc:
        print(f"❌ Błąd: {exc}")
    finally:
        sys.stdout = old_stdout
        q.put(None)  # sygnał końca


def launch_pipeline(uploaded_file):
    """Generator – yielduje log na bieżąco i na końcu zwraca pliki."""
    if uploaded_file is None:
        yield "❌ Nie wybrano pliku.", None
        return

    q: queue.Queue = queue.Queue()
    t = threading.Thread(target=run_pipeline_thread, args=(uploaded_file, q), daemon=True)
    t.start()

    log_lines = []
    output_files = None

    while True:
        item = q.get()

        if item is None:                        # koniec
            break
        if isinstance(item, tuple) and item[0] == "__files__":
            output_files = item[1]
            continue

        log_lines.append(item)
        yield "\n".join(log_lines), output_files  # ← aktualizacja UI po każdej linii

    t.join()
    yield "\n".join(log_lines), output_files      # ostateczny stan


# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Pipeline",
    theme=gr.themes.Base(
        primary_hue="indigo",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("IBM Plex Mono"),
    ),
    css="""
        #log_box textarea {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.85rem;
            background: #0f172a;
            color: #94a3b8;
            border: 1px solid #1e293b;
            border-radius: 8px;
        }
    """,
) as demo:

    gr.Markdown("# ⚙️ Pipeline Runner\nWgraj plik `.txt`, uruchom przetwarzanie i pobierz wyniki.")

    file_input = gr.File(label="📂 Plik wejściowy (.txt)", file_types=[".txt"], type="filepath")
    run_btn    = gr.Button("▶ Uruchom", variant="primary")
    log_output = gr.Textbox(label="📋 Log", lines=20, interactive=False, elem_id="log_box")
    dl_output  = gr.File(label="⬇️ Pobierz wyniki", interactive=False, file_count="multiple")

    run_btn.click(
        fn=launch_pipeline,
        inputs=[file_input],
        outputs=[log_output, dl_output],
    )

if __name__ == "__main__":
    demo.queue().launch()   # .queue() jest wymagane dla generatorów