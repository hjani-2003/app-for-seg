"""Runs PyRadiomics over a batch of image/mask pairs, in PyRadiomics' own env.

This file is executed by the radiomics_39 interpreter, not by the app — see
core/backends/radiomics_backend.py for why they cannot be the same one. It must
therefore stay importable on Python 3.9 and depend on nothing but the standard
library and `radiomics` itself.

It takes the place of the `pyradiomics` CLI, which can do batches, because a
batch there is all-or-nothing: one region too small for a texture matrix aborts
the lot. Here each pair is isolated, so a degenerate ET region costs you that
row and nothing else. It also lets the parent stream progress, and hands back
JSON rather than a CSV so column order is the parent's decision.

Usage: python radiomics_runner.py <spec.json>
"""
import json
import logging
import sys


def _jsonable(value):
    """PyRadiomics returns numpy scalars and arrays; JSON knows neither."""
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def main(spec_path):
    with open(spec_path) as handle:
        spec = json.load(handle)

    # PyRadiomics is chatty — it narrates each feature class, and warns about
    # identities like "Sum Average = 2 * Joint Average" on every single ROI.
    # Real problems are raised, not logged, so the output handler is turned
    # down to errors. This has to happen after the import: radiomics/__init__
    # sets the level itself, so quieting the logger beforehand is undone.
    import radiomics
    from radiomics import featureextractor

    radiomics.setVerbosity(logging.ERROR)

    extractor = featureextractor.RadiomicsFeatureExtractor(spec["params"])

    rows = []
    skipped = []
    jobs = spec["jobs"]
    for index, job in enumerate(jobs, start=1):
        # The parent parses this off stdout to drive the status bar.
        print(
            "PROGRESS %d/%d %s %s" % (index, len(jobs), job["modality"], job["region"]),
            flush=True,
        )
        try:
            result = extractor.execute(job["image"], job["mask"], label=1)
        except Exception as exc:
            skipped.append([job["modality"], job["region"], str(exc)])
            continue

        features = {}
        for key, value in result.items():
            if key.startswith("diagnostics_"):
                continue
            features[key] = _jsonable(value)

        rows.append(
            {
                "modality": job["modality"],
                "region": job["region"],
                "features": features,
            }
        )

    with open(spec["output"], "w") as handle:
        json.dump(
            {"version": radiomics.__version__, "rows": rows, "skipped": skipped},
            handle,
        )


if __name__ == "__main__":
    main(sys.argv[1])
