import asyncio
import json
from typing import List, Dict, Any
from datetime import datetime

from app.graph.state import State
from app.graph.graph import graph

class DateTimeEncoder(json.JSONEncoder):
    """Custom encoder który radzi sobie z datetime i innymi typami"""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)


async def process_candidates_with_batching(
    candidates: List[Dict],
    max_concurrent: int = 15,
    batch_size: int = 20
) -> List[Dict[str, Any]]:
    """
    Przetwarza kandydatów z kontrolą współbieżności i batchingiem.
    
    Args:
        candidates: Lista kandydatów do przetworzenia
        max_concurrent: Maksymalna liczba równoległych grafów
        batch_size: Rozmiar batcha dla zapisu wyników
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    
    async def process_with_semaphore(candidate: Dict, index: int) -> Dict[str, Any]:
        """Wrapper który kontroluje współbieżność"""
        async with semaphore:
            try:
                result: State = await graph.ainvoke({"message": candidate})
                return {
                    "index": index,
                    "candidate": candidate,
                    "result": result,
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                print(f"❌ Błąd dla kandydata {index + 1}: {str(e)}")
                return {
                    "index": index,
                    "candidate": candidate,
                    "error": str(e),
                    "status": "error",
                    "timestamp": datetime.now().isoformat()
                }
    
    tasks = [
        process_with_semaphore(candidate, i) 
        for i, candidate in enumerate(candidates)
    ]
    
    for i, coro in enumerate(asyncio.as_completed(tasks)):
        result = await coro
        results.append(result)
        
        # Zapis częściowy co batch_size elementów (do pliku partial, nie finalnego)
        if (i + 1) % batch_size == 0:
            save_results_to_json(results, partial=True, partial_index=i + 1)
            print(f"Zakończono analizę {i + 1} postów")
    
    # Zapis finalny po przetworzeniu wszystkich kandydatów
    save_results_to_json(results, partial=False)
    print(f"✅ Zapisano finalny wynik dla wszystkich {len(results)} kandydatów")

    return results

def save_results_to_json(
    results: List[Dict],
    partial: bool = False,
    partial_index: int = 0
):
    """
    Zapisuje wyniki do plików JSON na podstawie is_lead oraz błędów.
    
    Pliki finalne (partial=False):
        lead_YYYY-MM-DD.json
        no_lead_YYYY-MM-DD.json
        errors_YYYY-MM-DD.json
    
    Pliki częściowe (partial=True) mają suffix _partial_{index}
    i nigdy nie nadpisują plików finalnych.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    
    if partial:
        files = {
            True:    f"lead_{today}_partial_{partial_index}.json",
            False:   f"no_lead_{today}_partial_{partial_index}.json",
            "error": f"errors_{today}_partial_{partial_index}.json",
        }
    else:
        files = {
            True:    f"lead_{today}.json",
            False:   f"no_lead_{today}.json",
            "error": f"errors_{today}.json",
        }
    
    grouped: dict = {True: [], False: [], "error": []}
    
    for r in results:
        if r.get("status") == "error":
            grouped["error"].append({
                "message": r.get("candidate"),
                "error": r.get("error"),
            })
            continue
        
        if r.get("status") != "success":
            continue
        
        result = r.get("result", {})
        lead_judge = result.get("lead_judge")
        if lead_judge is None:
            continue
        
        entry = {
            "original_message": result.get("message"),
            "message": result["message"].get("message"),
            "user": result["message"].get("user"),
            "is_lead": result["lead_judge"].is_lead,
            "rag_insight": result.get('rag_insight', None),
            "reply": result["reply"].reply,
        }
        grouped[lead_judge.is_lead].append(entry)

    for key, entries in grouped.items():
        if not entries:
            print(f"ℹ️ Brak wyników dla '{key}'. Pomijam zapis.")
            continue
        with open(files[key], "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2, cls=DateTimeEncoder)
        print(f"📁 Zapisano {len(entries)} wyników do {files[key]}")