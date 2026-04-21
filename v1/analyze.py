import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from inference import NeSyWareInference, PREDICATE_LABELS

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║                        ⚠  DISCLAIMER                        ║
╠══════════════════════════════════════════════════════════════╣
║  NeSyWare is an AI-based research prototype and may produce  ║
║  incorrect or incomplete classifications.                    ║
║                                                              ║
║  Always cross-validate results with:                         ║
║    • VirusTotal or equivalent multi-engine scanners          ║
║    • Static analysis tools (e.g. Ghidra, PE-bear)            ║
║    • Dynamic / sandbox analysis (e.g. ANY.RUN, Cuckoo)       ║
║                                                              ║
║  Symbolic predicates reflect visual correlates of behaviour  ║
║  — not verified runtime actions.                             ║
║                                                              ║
║  Do not use as a sole basis for security decisions.          ║
╚══════════════════════════════════════════════════════════════╝
""")

    if len(sys.argv) < 2:
        print("Usage: python analyze.py <file> [file2 ...]")
        sys.exit(1)

    engine = NeSyWareInference()
    engine.load(progress_callback=lambda m: print(f"[*] {m}"))

    for path in sys.argv[1:]:
        print(f"\n{'='*60}")
        print(f"File: {path}")
        result = engine.analyze(path)

        if result["error"]:
            print(f"ERROR: {result['error'][:200]}")
            continue

        label = result["family_profile_label"].upper()
        fam   = result["family"] or "inconclusive"
        conf  = (result["family_conf"] or 0.0) * 100
        cat   = result["category"] or "—"

        print(f"Result  : [{label}] {fam}  ({conf:.1f}%)  [{cat}]")

        if result["family_profile"]:
            print("Profile :")
            for rank, (f, c) in enumerate(result["family_profile"], 1):
                bar = "█" * int(c * 40)
                print(f"  {rank:2}. {f:<22} {bar:<40} {c*100:5.1f}%")

        active = result["active_predicates"]
        if active:
            print("Predicates (active):")
            for pname, pval in active:
                label_str = PREDICATE_LABELS.get(pname, pname)
                print(f"  {label_str:<30} {pval:.2f}")

if __name__ == "__main__":
    main()