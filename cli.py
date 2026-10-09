"""Interactive command-line interface for the document Q&A bot."""
from rag_pipeline import answer_question, DEFAULT_TOP_K

def main() -> None:
    print("Bookxpert Basic Document Q&A Bot")
    print("Type a question, or type 'exit' to quit.")
    while True:
        try:
            question = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye.")
            break
        if not question:
            continue
        try:
            answer, sources = answer_question(question, top_k=DEFAULT_TOP_K)
            print(f"\nAnswer:\n{answer}")
            print("\nRetrieved sources:")
            for i, source in enumerate(sources, start=1):
                location = f" — page {source['page']}" if source.get("page") else ""
                print(f"\n[{i}] {source['source']}{location} (similarity={source['score']:.3f})")
                print(source["text"][:700] + ("..." if len(source["text"]) > 700 else ""))
        except Exception as exc:
            print(f"\nCould not answer the question: {exc}")

if __name__ == "__main__":
    main()
