import asyncio

from app.regex_check import process_messages
from utils.process_graphs import process_candidates_with_batching

async def main():
    print("Start...")
    try:
        with open("treść2.txt", "r", encoding="utf-8") as f:
            text = f.read()
            candidates, all_messages = process_messages(text)
            results = await process_candidates_with_batching(
                candidates,
                max_concurrent=15,  
                batch_size=20
            )
            
            # Podsumowanie
            successful = [r for r in results if r["status"] == "success"]
            failed = [r for r in results if r["status"] == "error"]
            
            print(f"\n📈 Podsumowanie:")
            print(f"   ✅ Sukces: {len(successful)}")
            print(f"   ❌ Błędy: {len(failed)}")
            
    except FileNotFoundError:
        print("❌ Nie znaleziono pliku 'treść1.txt'")
        return []

if __name__ == "__main__":
    asyncio.run(main())
