"""TheGuard agentic layer.

Takes the ML classifier's risk score and turns it into a human explanation:
which red flags fired, why each one is dangerous, what to do next, in
English and Hinglish. This is the 'agentic AI' component of the project -
the model scores, the agent reasons, explains and acts.
"""
import re

RED_FLAG_RULES = [
    ("Suspicious link",
     r"https?://|bit\.ly|tinyurl|shorturl|\.top\b|\.xyz\b|\.online\b|\.apk\b",
     "The message pushes you to a link. Genuine banks do not send KYC or bill links on SMS. "
     "Shortened and cheap domains (.top, .xyz) are used to hide the real website."),
    ("Asks for OTP / PIN / CVV",
     r"\botp\b|\bpin\b|\bcvv\b|\bpassword\b|\bcode sent\b|share the code|read the otp",
     "No bank, company or government office will ever ask for your OTP, PIN, CVV or password. "
     "Anyone who asks is a fraudster, without exception."),
    ("Account / service threat",
     r"block(ed)?|suspend(ed)?|deactivat|disconnect|frozen|deactivat|cancel(led)?|banned|closed|penalty|fine|arrest|warrant|legal action|jail|court",
     "Fear and a deadline is the oldest trick in the book. Real institutions send written notice "
     "and give you time - they never threaten instant blocking in a text message."),
    ("Urgency pressure",
     r"immediately|urgent|today|tonight|within \d+ hour|last warning|last chance|final notice|expire|before midnight|24 hours|2 hours",
     "Manufactured urgency stops you from thinking or checking with someone. "
     "Slow down - that alone defeats most of these attacks."),
    ("Money transfer demanded",
     r"\bpay\b|payment|transfer|deposit|processing fee|registration fee|clearance|settlement|refund|remit|send rs|scan this qr|upi id|@ybl|@paytm|@ok",
     "Money moving out is the whole point of the scam. Genuine refunds and prizes never "
     "require you to pay first or to scan a QR code."),
    ("Prize / lottery bait",
     r"congratulat|winner|won|lucky draw|lottery|prize|selected|free (iphone|gift|laptop|sim|recharge)|gift card|cashback",
     "You cannot win a lottery you never entered. 'Pay a small fee to claim a big prize' "
     "is a pure extraction scam."),
    ("Impersonates bank / government",
     r"\bkyc\b|\bsbi\b|\bhdfc\b|\bicici\b|\baxis\b|\bpaytm\b|\bphonepe\b|\bgpay\b|\bupi\b|income tax|gst|itr|cbi|police|cyber cell|\bed\b|trai|customs|india post|electricity|aadhaar|pan card|ration",
     "The sender is copying a trusted name to borrow its authority. Check the actual sender ID "
     "and contact the institution on the number printed on your card or bill."),
    ("Too-good-to-be-true income",
     r"earn \d+|per day|daily income|work from home|part time job|guaranteed|double your money|no risk|profit|investment|trading",
     "Guaranteed high returns do not exist. Work-from-home 'joining fee' and telegram "
     "trading groups are among the fastest growing fraud categories in India."),
    ("Remote access / unknown app",
     r"anydesk|teamviewer|quick ?support|install this apk|download this apk|screen share|remote",
     "Remote-access apps hand your entire phone to the caller, including your banking app. "
     "Never install one for a 'support' call you did not initiate."),
    ("Personal data harvesting",
     r"aadhaar|pan card|card number|expiry|cvv|bank account number|ifsc|send your .*photo|share your details|date of birth",
     "Your ID numbers and card photos are enough to open accounts and loans in your name. "
     "They are never needed over SMS or WhatsApp."),
]

NEXT_STEPS = [
    "Do not click any link, do not scan any QR code, and do not reply to the message.",
    "Do not share the OTP, PIN, CVV or password with anyone - not even someone claiming to be bank staff.",
    "Verify on your own: open the bank's official app or call the number printed on your card / bill.",
    "If money has already left your account, call the national cyber crime helpline 1930 immediately - the first 24 hours matter most.",
    "Report the message at cybercrime.gov.in and forward it to 1909 to report the sender.",
    "Tell one other person - a parent, neighbour or colleague - what this message looked like.",
]

NEXT_STEPS_HI = [
    "Kisi bhi link par click na karein, koi QR code scan na karein, aur message ka jawab na dein.",
    "OTP, PIN, CVV ya password kisi ke saath share na karein - chahe woh khud ko bank staff bole.",
    "Khud verify karein: bank ka official app kholein ya card / bill par likhe number par call karein.",
    "Agar paisa nikal gaya hai to turant national cyber crime helpline 1930 par call karein - pehle 24 ghante sabse zaroori hain.",
    "Message ko cybercrime.gov.in par report karein aur 1909 par forward karein.",
    "Kisi ek aur insaan ko batayein - parents, padosi ya colleague - taaki woh bhi bach sakein.",
]


def extract_red_flags(message: str):
    """Return the list of red flags the agent found, with the evidence text."""
    found = []
    low = message.lower()
    for label, pattern, why in RED_FLAG_RULES:
        m = re.search(pattern, low, flags=re.IGNORECASE)
        if m:
            start = max(0, m.start() - 25)
            end = min(len(message), m.end() + 25)
            found.append({
                "flag": label,
                "evidence": message[start:end].strip(),
                "why": why,
            })
    return found


def decide(score: float, n_flags: int):
    """Blend the ML score with the agent's rule evidence into a verdict."""
    blended = min(1.0, score + (0.03 * min(n_flags, 4)))
    if blended >= 0.75:
        return "SCAM", blended
    if blended >= 0.45:
        return "SUSPICIOUS", blended
    if blended >= 0.25:
        return "BE CAREFUL", blended
    return "LOOKS SAFE", blended


def explain(message: str, score: float, lang: str = "en"):
    """Full agent output: verdict, reasoning trace, advice, next steps."""
    flags = extract_red_flags(message)
    verdict, blended = decide(score, len(flags))

    if verdict == "SCAM":
        headline = ("This message is almost certainly a fraud attempt. Do not act on it.")
        headline_hi = "Yeh message lagbhag pakka fraud hai. Is par koi action na lein."
    elif verdict == "SUSPICIOUS":
        headline = ("This message shows real signs of fraud. Treat it as unsafe until you verify it yourself.")
        headline_hi = "Is message mein fraud ke saaf sanket hain. Jab tak khud verify na karein, ise unsafe maanein."
    elif verdict == "BE CAREFUL":
        headline = ("This message looks mostly ordinary but has one or two things worth checking.")
        headline_hi = "Yeh message aam lagta hai, lekin ek-do cheezein check karne layak hain."
    else:
        headline = ("Nothing in this message matches known fraud patterns. Stay alert anyway.")
        headline_hi = "Is message mein fraud ka koi jaana-pehchaana pattern nahi mila. Phir bhi saavdhan rahein."

    trace = [
        f"Model scored the message at {score*100:.1f}% fraud probability.",
        f"Agent matched {len(flags)} of {len(RED_FLAG_RULES)} known fraud red-flag rules.",
        f"Combined verdict: {verdict} at {blended*100:.1f}% confidence.",
    ]
    if flags:
        trace.append("Strongest signals: " + ", ".join(f["flag"] for f in flags[:3]) + ".")
    else:
        trace.append("No link, OTP request, threat, payment demand or prize bait was detected.")

    return {
        "verdict": verdict,
        "confidence": round(blended, 4),
        "model_score": round(score, 4),
        "headline": headline,
        "headline_hi": headline_hi,
        "red_flags": flags,
        "reasoning_trace": trace,
        "next_steps": NEXT_STEPS if lang == "en" else NEXT_STEPS_HI,
        "helpline": {"number": "1930", "portal": "cybercrime.gov.in", "report_sms": "1909"},
    }
