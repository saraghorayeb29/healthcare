"""Command line: `pdac-panel fetch` downloads the data, `pdac-panel run` writes the results."""
import argparse
import json
import pathlib

from . import analysis, data


def main():
    parser = argparse.ArgumentParser(prog="pdac-panel")
    parser.add_argument("command", choices=["fetch", "run"])
    parser.add_argument("--out", default="results", type=pathlib.Path)
    args = parser.parse_args()
    if args.command == "fetch":
        data.fetch()
        return
    results, panels = analysis.run()
    args.out.mkdir(exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    panels.round(4).to_csv(args.out / "panels.csv", index=False)
    cohort, chosen = results["cohort"], results["selected"]
    print(f"{cohort['cases']} cases, {cohort['controls']} controls, {results['panel_count']} panels, {results['model_count']} models")
    for row in results["anova"]:
        print(f"  {row['biomarker']:7} F={row['F']:8.1f} p={row['p']:.2e} {'selected' if row['selected'] else 'rejected'}")
    print("panels that pass:", results["passing"])
    print("selected:", chosen["panel"], "| passes all three criteria:", chosen["passes"])
    for row in results["comparison"]:
        print(f"  {row['label']:24} {row['panel']:40} AUC {row['auc']:.3f} [{row['ci'][0]:.3f}, {row['ci'][1]:.3f}]"
              f"  detects {row['true_positive']}/{row['cases']}, false positives {row['false_positive']}/{row['controls']}")
    for label, row in results["differences"].items():
        print(f"  {label}: {row['difference']:+.3f} [{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
