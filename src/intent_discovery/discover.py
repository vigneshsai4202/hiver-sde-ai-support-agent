import re
from collections import Counter

KEYWORDS = {
 "IOS_UPDATE":[r"\bios\b",r"update",r"upgrade"],
 "BATTERY_POWER":[r"battery",r"charging",r"charger",r"charge"],
 "CONNECTIVITY":[r"wifi",r"wi-fi",r"bluetooth",r"network",r"signal"],
 "APPLE_ID_ICLOUD":[r"apple id",r"icloud",r"sign in",r"login"],
 "DEVICE_PERFORMANCE":[r"crash",r"freeze",r"restart",r"slow",r"lag"],
 "APPS_APP_STORE":[r"app store",r"app ",r"download"],
 "MEDIA_SERVICES":[r"apple music",r"itunes",r"podcast"],
 "PAYMENTS_PURCHASES":[r"charge",r"charged",r"refund",r"payment",r"purchase"],
 "DATA_BACKUP_RESTORE":[r"backup",r"restore",r"recover",r"sync"]
}

def heuristic_intent(text):
    scores = Counter()
    for intent, patterns in KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text.lower()):
                scores[intent] += 1
    return scores.most_common(1)[0][0] if scores else "OTHER"
