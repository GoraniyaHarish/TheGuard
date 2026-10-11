"""TheGuard - training and honest evaluation.

Trains four detectors and compares them fairly:

    baseline   TF-IDF (word+char) + LogReg          (the original model)
    hybrid     TF-IDF + engineered signals + LogReg (explainable reasoning)
    embeddings pretrained sentence-transformer + LogReg
    ensemble   average of hybrid + embeddings

Every model is scored with:
  * 5-fold stratified cross-validation on the full corpus (out-of-fold), and
  * an unseen, adversarial probe set (realistic cases a shallow model misses).

The best model is written to model.joblib (same predict_proba API the Flask app
already uses) and a full report to metrics.json.
"""
import json
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_auc_score)

import models
from features import FEATURE_NAMES, N_ENGINEERED, extract, SIGNAL_TEXT
from dataset import load_all, PROBE

RNG = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)


def score(y_true, p):
    """Metrics from probabilities at the 0.5 threshold."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(p) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, p)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def oof(est, X, y):
    return cross_val_predict(est, X, y, cv=CV, method="predict_proba")[:, 1]


def print_metrics(tag, m):
    print(f"  {tag:<11} acc={m['accuracy']:.3f}  prec={m['precision']:.3f}  "
          f"rec={m['recall']:.3f}  f1={m['f1']:.3f}  auc={m['roc_auc']:.3f}")


def main():
    print("=" * 68)
    X, y, meta = load_all()
    print("Corpus sources:")
    for k, v in meta.items():
        print(f"  {k:<16} {v}")
    y = np.asarray(y)

    dets = {}          # name -> sklearn estimator (for final fit)
    oofs = {}          # name -> out-of-fold probabilities

    # ---- baseline & hybrid (pure sklearn, cross-validated directly) --------
    dets["baseline"] = models.baseline()
    dets["hybrid"] = models.hybrid()
    for name in ("baseline", "hybrid"):
        oofs[name] = oof(dets[name], X, y)

    # ---- transformer embeddings -------------------------------------------
    try:
        from sentence_transformers import SentenceTransformer
        from models import DEFAULT_ST_MODEL
        print(f"\nLoading sentence-transformer: {DEFAULT_ST_MODEL}")
        st = SentenceTransformer(DEFAULT_ST_MODEL)
        E = st.encode(list(X), batch_size=64, normalize_embeddings=True,
                      show_progress_bar=False)
        E = np.asarray(E)
        emb_clf_est = LogisticRegression(max_iter=3000, C=8.0, class_weight="balanced")
        oofs["embeddings"] = cross_val_predict(emb_clf_est, E, y, cv=CV,
                                               method="predict_proba")[:, 1]
        dets["embeddings"] = models.embeddings()   # fresh pipeline for the final fit
        print(f"  embeddings: {E.shape[1]}-dim vectors")


    except Exception as exc:  # noqa: BLE001
        print(f"\n[warn] embeddings unavailable ({exc}); skipping transformer model")

    # ---- ensemble of hybrid + embeddings ----------------------------------
    if "embeddings" in oofs:
        oofs["ensemble"] = 0.5 * oofs["hybrid"] + 0.5 * oofs["embeddings"]

    # ---- cross-validated report -------------------------------------------
    print("\n=== 5-fold cross-validation (out-of-fold, full corpus) ===")
    cv_report = {}
    for name, p in oofs.items():
        cv_report[name] = score(y, p)
        print_metrics(name, cv_report[name])

    # ---- adversarial probe report (models refit on everything, probe unseen)
    print("\n=== Adversarial probe set (unseen, realistic hard cases) ===")
    ptexts = [t for t, _ in PROBE]
    py = np.array([lbl for _, lbl in PROBE])
    probe_report = {}
    fitted = {}
    for name, est in dets.items():
        try:
            est.fit(X, y)
            fitted[name] = est
            p = est.predict_proba(ptexts)[:, 1]
            probe_report[name] = score(py, p)
            print_metrics(name, probe_report[name])
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {name} failed on probe: {exc}")

    if "hybrid" in fitted and "embeddings" in fitted:
        ens = models.ensemble([fitted["hybrid"], fitted["embeddings"]])
        fitted["ensemble"] = ens
        p = ens.predict_proba(ptexts)[:, 1]
        probe_report["ensemble"] = score(py, p)
        print_metrics("ensemble", probe_report["ensemble"])

    # ---- pick the best model (realism-weighted) ---------------------------
    print("\n=== Model selection (0.4*CV f1 + 0.6*probe accuracy) ===")
    best_name, best_score = None, -1
    for name in oofs:
        if name not in probe_report:
            continue
        s = 0.4 * cv_report[name]["f1"] + 0.6 * probe_report[name]["accuracy"]
        print(f"  {name:<11} score={s:.4f}")
        if s > best_score:
            best_name, best_score = name, s
    print(f"\nSelected model: {best_name}")

    # ---- show exactly which hard cases the winner still gets wrong --------
    if best_name in fitted:
        pp = fitted[best_name].predict_proba(ptexts)[:, 1]
        wrong = [((t, lbl), float(p)) for (t, lbl), p in zip(PROBE, pp)
                 if (p >= 0.5) != bool(lbl)]
        print(f"\nProbe errors for {best_name}: {len(wrong)}/{len(PROBE)}")
        for (t, lbl), p in wrong:
            print(f"  true={'SCAM' if lbl else 'SAFE'}  p={p:.2f}  {t[:72]}")

    # ---- engineered-signal importances (why the hybrid reasons) -----------
    signal_importance = []
    if "hybrid" in fitted:
        coef = fitted["hybrid"].named_steps["clf"].coef_[0][-N_ENGINEERED:]
        for nm, c in sorted(zip(FEATURE_NAMES, coef), key=lambda kv: -abs(kv[1])):
            signal_importance.append({"signal": nm, "description": SIGNAL_TEXT.get(nm, nm),
                                      "coefficient": round(float(c), 3)})

    # ---- persist the winner ----------------------------------------------
    joblib.dump(fitted[best_name], "model.joblib")

    report = {
        "chosen_model": best_name,
        "models": {
            "baseline": "TF-IDF(word 1-2 + char_wb 3-5) + LogisticRegression",
            "hybrid": "TF-IDF + engineered scam-signal features + LogisticRegression",
            "embeddings": "sentence-transformers/all-MiniLM-L6-v2 + LogisticRegression",
            "ensemble": "mean(hybrid, embeddings)",
        },
        "corpus": meta,
        "signals": FEATURE_NAMES,
        "cross_validation": {"folds": 5, "results": cv_report},
        "probe": {"n": len(PROBE), "results": probe_report},
        "signal_importance": signal_importance,
    }
    with open("metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\nSaved model.joblib and metrics.json")
    print("=" * 68)


if __name__ == "__main__":
    main()
