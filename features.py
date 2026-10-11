"""TheGuard engineered scam-signal features.

The original model only saw raw TF-IDF tokens, so it learned surface words
("http", "pay", "now") instead of the *reasons* a message is a scam. This
module turns each message into an explicit, human-meaningful vector of signals
(a link to a shortened/cheap domain, a request for OTP, manufactured urgency,
a threat, a money demand, impersonation of a bank, remote-access bait, etc.).

These features are:
  * interpretable - every column has a name a human can read,
  * robust - they generalise to unseen wording, and
  * complementary - they are combined with TF-IDF in models.py.
"""
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------
# vocabularies / patterns
# --------------------------------------------------------------------------
SHORTENERS = ("bit.ly", "tinyurl", "t.co", "goo.gl", "ow.ly", "is.gd", "rb.gy",
              "cutt.ly", "shorturl", "rebrand.ly", "buff.ly", "tiny.cc", "adf.ly")
SUSP_TLDS = (".top", ".xyz", ".online", ".club", ".buzz", ".click", ".link",
             ".live", ".icu", ".work", ".rest", ".shop", ".site", ".website",
             ".fun", ".monster", ".loan", ".download", ".stream", ".gq", ".cf", ".tk")
BAIT_DOMAIN_WORDS = ("kyc", "verify", "update", "login", "pay", "secure", "reward",
                     "refund", "winner", "offer", "claim", "account", "lock", "unlock",
                     "restore", "help", "support", "bill", "ekyc", "appeal")
BRANDS = ("sbi", "hdfc", "icici", "axis", "kotak", "pnb", "bob", "canara", "unionbank",
          "paytm", "phonepe", "gpay", "googlepay", "upi", "amazon", "flipkart", "netflix",
          "apple", "icloud", "google", "whatsapp", "instagram", "facebook", "jio", "airtel",
          "vodafone", "vi", "bsnl", "indiapost", "fedex", "dhl", "dtdc", "irctc", "sbi",
          "incometax", "gst", "epfo", "nsdl", "ayushman", "pmkisan", "vahan", "fastag")

URL_RE = re.compile(r"(https?://\S+|www\.\S+|\S+\.(?:com|in|net|org|co|top|xyz|online|club|info|link|site|shop|icu|buzz|icu|apk)(?:/\S*)?)", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\-\s]?)?[6-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
AMOUNT_RE = re.compile(r"(?:rs\.?|inr|₹)\s?([\d][\d,]*(?:\.\d+)?)", re.I)
LAKH_RE = re.compile(r"(\d[\d,]*)\s?(lakh|lac|crore|cr\b)", re.I)
IP_URL_RE = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")

# behavioural signal vocabularies (matched against lowercased text)
SIG = {
    "otp_request": r"share the (code|otp|pin)|read (the|out) the? ?otp|tell me the \d+ digit|"
                   r"give me the (code|otp)|send (me )?the (code|otp)|forward the (code|otp)|"
                   r"confirm the otp|provide the otp|what is the otp|share your (otp|pin|cvv|password)|"
                   r"otp.*(to|for) (confirm|verify|update|complete|reactivate)|read the otp you|"
                   r"share (it|the code) with our (executive|staff|officer)|code sent to your (mobile|number)",
    "otp_protective": r"do not (share|disclose|reveal)|never (share|disclose|reveal)|"
                      r"not share (this|the) (code|otp|pin)|do not share (this )?(code|otp|pin)|"
                      r"keep (this|it) (code|otp|pin)? ?(safe|secret|confidential)|"
                      r"do not share it|not even with bank staff|will never ask",
    "safety_advice": r"will never (ask|call)|never ask for your (otp|pin|password)|"
                     r"beware|stay (alert|safe)|report (any )?(fraud|suspicious)|"
                     r"helpline 1930|national cyber|cyber ?crime|"
                     r"no fee is (charged|required)|no (payment|fee) (is )?(required|needed|charged)|"
                     r"no action (is )?(needed|required)|"
                     r"official\b[^.!?]{0,20}?\b(app|portal|website|site|discom|company|channel|bank)|"
                     r"visit the branch|call the number (on|printed)",
    "txn_alert": r"debited from (a/?c|your account|your card)|credited to (a/?c|your account)|"
                 r"has been credited|has been debited|available balance|remaining balance|"
                 r"available limit|balance rs|ref(erence)? ?(no\.?|number)? ?\d{6,}|"
                 r"upi ref|not you\?|is successful|was successful|"
                 r"a/?c xx?\d|account (ending|xx)|otp for the transaction of",
    "personal_info": r"aadhaar|aadhar|\bpan card\b|card number|expiry|debit card|credit card|"
                     r"bank account number|\bifsc\b|passbook|date of birth|\bdob\b|"
                     r"send your .*(photo|details)|share your (details|bank)|"
                     r"front and back|upi pin|net ?banking (user ?id|password)",
    "threat": r"block(ed|ing)?|suspend(ed|ing)?|deactivat|disconnect|frozen|freeze|"
              r"cancel(led|ing)?|banned|banned|closed|penalty|fine|arrest|warrant|"
              r"legal action|jail|court|police|cyber cell|\bcbi\b|\bed\b|notice|"
              r"terminat|blacklist|lawsuit|prosecut",
    "urgency": r"immediat|urgent|right now|\btoday\b|tonight|within \d+\s?(hour|hr|min|day)|"
               r"last warning|last chance|final notice|final warning|expir|before midnight|"
               r"24 hours|2 hours|48 hours|\basap\b|\bnow\b|hurry|limited (period|time|offer)|"
               r"act fast|only \d+ (left|remaining)|deadline",
    "money": r"\bpay\b|payment|transfer|deposit|processing fee|registration fee|"
             r"clearance (fee|charge)|settlement|refund|remit|send (rs|money|cash)|"
             r"upi id|@ybl|@paytm|@ok|scan (this )?qr|scan the qr|neft|imps|"
             r"remit|wire|advance (payment|fee)|security deposit|joining fee",
    "prize": r"congratulat|winner|\bwon\b|lucky draw|lottery|prize|you (are|have been) selected|"
             r"free (iphone|gift|laptop|sim|recharge|entry|data|voucher)|gift card|cashback|"
             r"claim your|jackpot|bumper",
    "impersonation": r"\bkyc\b|\bsbi\b|\bhdfc\b|\bicici\b|\baxis\b|\bpaytm\b|\bphonepe\b|"
                     r"\bgpay\b|\bupi\b|income tax|\bgst\b|\bitr\b|\bcbi\b|police|cyber cell|"
                     r"\bed\b|\btrai\b|customs|india post|electricity|water bill|\baadhaar\b|"
                     r"pan card|ration|government|\bpm\b|epfo|\bepf\b|vahan|fastag|"
                     r"customer care|support team|it department|bank (staff|official|officer)|"
                     r"from (your|the) bank|mobile network|network provider",
    "investment": r"earn \d+|per day|daily (income|profit|earning)|work from home|part[\s-]?time job|"
                  r"guaranteed|double your money|no risk|risk[\s-]?free|profit|investment|"
                  r"trading|forex|crypto|bitcoin|stock tips|sure shot|multiply your money",
    "remote": r"anydesk|team ?viewer|quick ?support|install (this )?apk|download (this )?apk|"
              r"screen shar|remote (access|support)|install (an|this) app|download the (app|apk)|"
              r"\.apk\b|join video call|video ?call now",
    "click_action": r"click (here|this|the link|below)|tap (here|this|the link)|"
                    r"visit (the )?(link|website|url)|open (the|this) link|follow (the|this) link|"
                    r"click link|check (the )?link|use the link",
    "verify_action": r"verif|update (your|the)|re-?kyc|complete (the |your )?(kyc|verification)|"
                     r"reactivat|un-?lock|unblock|restore|re-?activate|confirm (your|the)|"
                     r"validate|renew|submit (your|the) details|re-?register",
    "generic_greeting": r"^\s*(dear (customer|user|sir|madam|winner|candidate|member|consumer|"
                        r"employee|passenger|student|applicant|beneficiary|account holder|customer care)|"
                        r"dear valued|hello (dear|user|customer|winner)|congratulations)",
    "callback": r"call (this|the|our|on) ?(number|us|now|immediately|\+?\d)|"
                r"call back|toll ?free|helpline|\bcall\s?\d|dial \d|contact (us|this|the number)",
    "no_reply": r"do not (reply|call|contact)|please note|for queries|"
                r"customer care|unsubscribe|opt[- ]?out",
}
SIG_RE = {k: re.compile(v, re.I) for k, v in SIG.items()}

# for feature naming / explainability
FEATURE_NAMES = [
    "n_urls", "has_shortener", "has_susp_tld", "has_apk_link", "has_ip_url",
    "has_nonsecure_link", "has_bait_domain", "has_brand_bait_domain", "has_qr",
    "has_phone", "has_whatsapp_telegram", "has_email", "n_amounts", "max_amount_log",
    "has_big_money_word", "digit_ratio", "upper_ratio", "n_exclam_cap", "char_len_log",
    "word_count_log",
    "sig_otp_request", "sig_otp_protective", "sig_safety_advice", "sig_txn_alert",
    "sig_personal_info", "sig_threat", "sig_urgency", "sig_money",
    "sig_prize", "sig_impersonation", "sig_investment", "sig_remote",
    "sig_click_action", "sig_verify_action", "sig_generic_greeting", "sig_callback",
    "sig_no_reply",
]

N_ENGINEERED = len(FEATURE_NAMES)


def _hosts(text):
    hosts = []
    for u in URL_RE.findall(text):
        h = re.sub(r"^https?://", "", u, flags=re.I)
        h = re.sub(r"^www\.", "", h, flags=re.I)
        h = h.split("/")[0].split("?")[0].lower()
        if h:
            hosts.append(h)
    return hosts


def extract(text):
    """Return an ordered dict of engineered signals for one message."""
    t = text or ""
    low = t.lower()
    f = {}

    hosts = _hosts(t)
    domain_blob = " ".join(hosts)

    f["n_urls"] = min(len(hosts), 5)
    f["has_shortener"] = int(any(s in low for s in SHORTENERS))
    f["has_susp_tld"] = int(any(low.count(tld) for tld in SUSP_TLDS))
    f["has_apk_link"] = int(".apk" in low or ".exe" in low)
    f["has_ip_url"] = int(bool(IP_URL_RE.search(low)))
    f["has_nonsecure_link"] = int("http://" in low)
    f["has_bait_domain"] = int(any(w in domain_blob for w in BAIT_DOMAIN_WORDS) and bool(hosts))
    brand_bait = 0
    for h in hosts:
        has_brand = any(b in h for b in BRANDS)
        has_bait = any(w in h for w in BAIT_DOMAIN_WORDS)
        if has_brand and has_bait:
            brand_bait = 1
            break
    f["has_brand_bait_domain"] = brand_bait
    f["has_qr"] = int("qr code" in low or "scan the qr" in low or "scan this qr" in low or "qr scan" in low)

    f["has_phone"] = int(bool(PHONE_RE.search(low)))
    f["has_whatsapp_telegram"] = int(any(w in low for w in ("whatsapp", "telegram", "wa.me", "t.me", "chat on")))
    f["has_email"] = int(bool(EMAIL_RE.search(low)))

    amounts = []
    for m in AMOUNT_RE.finditer(t):
        try:
            amounts.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    for m in LAKH_RE.finditer(t):
        try:
            mult = 100000 if m.group(2).lower().startswith(("lakh", "lac")) else 10000000
            amounts.append(float(m.group(1).replace(",", "")) * mult)
        except ValueError:
            pass
    f["n_amounts"] = min(len(amounts), 5)
    f["max_amount_log"] = float(np.log1p(max(amounts))) if amounts else 0.0
    f["has_big_money_word"] = int(any(w in low for w in ("lakh", "crore", "lac ", "million", "prize money")))

    digits = sum(c.isdigit() for c in t)
    f["digit_ratio"] = digits / max(len(t), 1)
    letters = [c for c in t if c.isalpha()]
    f["upper_ratio"] = (sum(c.isupper() for c in letters) / len(letters)) if letters else 0.0
    f["n_exclam_cap"] = min(t.count("!") + t.count("!!"), 5)
    f["char_len_log"] = float(np.log1p(len(t)))
    f["word_count_log"] = float(np.log1p(len(t.split())))

    for key in ("otp_request", "otp_protective", "safety_advice", "txn_alert",
                "personal_info", "threat", "urgency", "money", "prize",
                "impersonation", "investment", "remote", "click_action",
                "verify_action", "generic_greeting", "callback", "no_reply"):
        f["sig_" + key] = int(bool(SIG_RE[key].search(low)))

    return f


def vector(text):
    d = extract(text)
    return np.array([d[name] for name in FEATURE_NAMES], dtype=np.float64)


# human-readable description of each engineered signal, for the UI/agent
SIGNAL_TEXT = {
    "n_urls": "Contains a link",
    "has_shortener": "Link uses a URL shortener (bit.ly/tinyurl-style) that hides the destination",
    "has_susp_tld": "Link uses a cheap/disposable domain (.top, .xyz, .online, .icu ...)",
    "has_apk_link": "Link points to an app file (.apk/.exe) - a classic way to plant malware",
    "has_ip_url": "Link uses a raw IP address instead of a real domain name",
    "has_nonsecure_link": "Link is plain http:// (not secure)",
    "has_bait_domain": "Link domain contains bait words (kyc, verify, login, pay, reward...)",
    "has_brand_bait_domain": "Link impersonates a brand (e.g. sbi-kyc, amazon-verify) but is not the official domain",
    "has_qr": "Asks you to scan a QR code",
    "has_phone": "Contains a phone number to call or reply to",
    "has_whatsapp_telegram": "Pushes you to WhatsApp/Telegram",
    "has_email": "Contains an email address",
    "n_amounts": "Mentions money amounts",
    "max_amount_log": "Mentions a large money amount",
    "has_big_money_word": "Talks about lakhs/crores/prize money",
    "digit_ratio": "Unusually digit-heavy text",
    "upper_ratio": "Unusually uppercase text (shouting/pressure)",
    "n_exclam_cap": "Uses exclamation marks to pressurise",
    "char_len_log": "Message length",
    "word_count_log": "Message length",
    "sig_otp_request": "Asks YOU to share/read the OTP, PIN, CVV or password",
    "sig_otp_protective": "Warns you NOT to share the code - a genuine safety message",
    "sig_safety_advice": "Gives official anti-fraud advice (use official app, contact branch, helpline 1930)",
    "sig_txn_alert": "Reads like a genuine bank transaction alert (balance, reference number, 'not you?')",
    "sig_personal_info": "Asks for personal identity data (Aadhaar/PAN/card details)",
    "sig_threat": "Threatens blocking, suspension, penalty, arrest or legal action",
    "sig_urgency": "Creates false urgency / a deadline",
    "sig_money": "Demands a payment, transfer or fee",
    "sig_prize": "Bait about a prize, lottery or 'free' gift",
    "sig_impersonation": "Impersonates a bank, government body or company",
    "sig_investment": "Promises guaranteed/high income or investment returns",
    "sig_remote": "Pushes remote-access or app installation",
    "sig_click_action": "Tells you to click a link",
    "sig_verify_action": "Tells you to verify/update/reactivate something",
    "sig_generic_greeting": "Uses a generic mass salutation (Dear customer/winner...)",
    "sig_callback": "Asks you to call a number / toll-free line",
    "sig_no_reply": "Attempts to sound official with 'do not reply / contact us' boilerplate",
}


def explain(text, top=6):
    """Return the fired signals as (label, description, value) tuples, strongest first."""
    d = extract(text)
    # protective signals argue for legitimacy, so they are not "red flags"
    negative = {"sig_otp_protective", "sig_safety_advice", "sig_txn_alert"}
    fired = []
    for name, val in d.items():
        if val > 0 and name not in negative and (
                name.startswith("sig_") or name.startswith("has_") or name == "n_urls"):
            fired.append((name, val))
    # simple priority: boolean behavioural signals first, then link, then numerics
    priority = {
        "sig_otp_request": 100, "sig_personal_info": 95, "has_apk_link": 92, "sig_remote": 90,
        "has_brand_bait_domain": 88, "has_shortener": 85, "sig_money": 80,
        "sig_threat": 78, "sig_impersonation": 76, "sig_urgency": 72,
        "sig_prize": 70, "sig_investment": 68, "has_susp_tld": 66,
        "has_bait_domain": 60, "has_qr": 58, "sig_callback": 50, "n_urls": 48,
    }
    fired.sort(key=lambda kv: (priority.get(kv[0], 10) - kv[1]), reverse=True)
    out = []
    for name, val in fired[:top]:
        out.append({"signal": name, "label": SIGNAL_TEXT.get(name, name), "value": val})
    return out


class SignalFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer: raw text list -> engineered signal matrix."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        return np.vstack([vector(t) for t in X])
