"""TheGuard data assembly.

Merges three sources into one labelled corpus (1 = fraudulent, 0 = legitimate):

  * the curated Indian-scam corpus in data.py (SCAM / HAM)
  * hard cases (HARD_SCAM / HARD_HAM) built to break shallow word matching
  * a large public real-world SMS corpus (datasets/sms_spam.tsv, ~5.5k messages)

The public corpus is the "reality check": it is not written by us, so it stops
the evaluation from being optimistic in the way a self-authored corpus is.
"""
import os
from data import SCAM, HAM, HARD_SCAM, HARD_HAM, TXN_ALERTS

BASE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_PATH = os.path.join(BASE, "datasets", "sms_spam.tsv")


def load_public(path=PUBLIC_PATH):
    """Load the UCI / pydata SMS Spam Collection (label<TAB>text)."""
    X, y = [], []
    if not os.path.exists(path):
        print(f"  [warn] public corpus not found at {path} - skipping")
        return X, y
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            text = text.strip()
            if not text:
                continue
            X.append(text)
            y.append(1 if label.strip().lower() == "spam" else 0)
    return X, y


def load_all(include_public=True):
    """Returns (X, y, meta). meta records how many came from each source."""
    X, y = [], []

    X += SCAM;       y += [1] * len(SCAM)
    X += HAM;        y += [0] * len(HAM)
    X += HARD_SCAM;  y += [1] * len(HARD_SCAM)
    X += HARD_HAM;   y += [0] * len(HARD_HAM)
    X += TXN_ALERTS; y += [0] * len(TXN_ALERTS)

    meta = {"curated_fraud": len(SCAM), "curated_legit": len(HAM),
            "hard_fraud": len(HARD_SCAM), "hard_legit": len(HARD_HAM),
            "txn_legit": len(TXN_ALERTS)}

    if include_public:
        px, py = load_public()
        X += px; y += py
        meta["public_fraud"] = int(sum(py))
        meta["public_legit"] = int(len(py) - sum(py))

    meta["total"] = len(X)
    meta["total_fraud"] = int(sum(y))
    meta["total_legit"] = int(len(y) - sum(y))
    return X, y, meta


# ---------------------------------------------------------------------------
# Realistic probe set: hand-picked cases that a shallow model gets wrong.
# Kept OUT of training so it is a fair, adversarial check.
# ---------------------------------------------------------------------------
PROBE = [
    # (text, label)  label 1 = scam, 0 = legitimate
    ("Your order is out for delivery. Track it live in the Amazon app.", 0),
    ("Amazon: your delivery is arriving today. Track at https://www.amazon.in/gp/order/123", 0),
    ("Your OTP for net banking login is 318204. Do not share it with anyone.", 0),
    ("Use the link to join the meeting: https://meet.google.com/xyz-abc-uvw", 0),
    ("Your electricity bill is ready. Pay via the official discom app or portal.", 0),
    ("Congratulations, you have been selected for the final interview round on Monday.", 0),
    ("You won the office quiz. Prize distribution is in the cafeteria at 5pm.", 0),
    ("Your refund of Rs 450 has been credited to the original payment method.", 0),
    ("A new device signed in to your account. If this was not you, secure it in settings.", 0),
    ("Your KYC has been verified at the branch. No further action is needed.", 0),
    ("The bank will never ask for your OTP or password. Report fraud to 1930.", 0),
    ("Sir main aapke bank se bol raha hoon, account block ho jayega, KYC ke liye call karein.", 1),
    ("Aapka number lottery me select hua hai, prize ke liye 5000 bhejein.", 1),
    ("Hello I am from bank IT department. Read the OTP you received to update your account.", 1),
    ("Please share your UPI pin to reverse a wrong transfer of Rs 15000.", 1),
    ("Your account shows suspicious activity. Confirm your card number and expiry by reply.", 1),
    ("Your aadhaar is linked to a crime. Pay a fine now to avoid a police case.", 1),
    ("Last warning. Your number will be deactivated. Complete e-KYC on this call.", 1),
    ("Dear user, your account is on hold. Verify your details at the link to restore it.", 1),
    ("Work from home and earn 3000 daily. Pay a small registration fee to start.", 1),
    ("Congratulations, you are today's lucky winner. Send your bank details.", 1),
    ("Your courier is stuck. Pay a small redelivery charge to release it.", 1),
    ("I mistakenly sent you money on UPI. Please return it quickly, this is urgent.", 1),
    ("Call this toll free number immediately, your computer has a virus.", 1),
    ("Your SIM will be blocked as per a new government rule. Complete verification now.", 1),
    ("Your job application is shortlisted. Pay the security deposit to confirm.", 1),
    ("I am an army officer. I will pay in advance via this link.", 1),
    ("Your exam result is out. Pay Rs 200 to unlock your marksheet.", 1),
    ("Your electricity will be disconnected tonight. Call this number to pay.", 1),
    ("A free 5G upgrade is available. Pay only Rs 99 for delivery.", 1),
    ("Your account has been suspended. Re-verify to reactivate it immediately.", 1),
    ("Dear sir we are updating our records. Confirm your bank account and IFSC by reply.", 1),
    ("Guaranteed 40 percent returns in 7 days. Invest a minimum of 5000 now.", 1),
    ("Please download AnyDesk and share your screen so we can fix your account.", 1),
    ("We noticed a login from a new device. Change your password in settings if this was not you.", 0),
    ("Your food order is on the way. The delivery partner will call you.", 0),
    ("Please click here to view your electricity bill on the official portal.", 0),
    ("Click the link to reset your password: https://accounts.example.com/reset", 0),
    ("Rs 780 debited from A/c XX3344 on 06-10-26 to a grocery store. Balance Rs 6100. Call 18001234567 if not you.", 0),
    ("IMPS of Rs 15000 to A/c XX7788 is successful on 07-10-26. Reference 889900112233.", 0),
    ("Your card XX1122 was used for an online payment of Rs 999. Available limit Rs 42000.", 0),
    ("A debit of Rs 8999 is pending on your card. Call this number immediately to stop it.", 1),
    ("Rs 9999 will be debited from your account in 1 hour. Call now to cancel it.", 1),
]
