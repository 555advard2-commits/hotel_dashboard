FAST_EFFECT_SUBJECTS = {
    "Rates improvement",
    "Special offer",
    "Non-refundable rate",
    "Rate without meals",
    "GEO-rates",
    "Parity",
    "Parity Yandex",
    "Parity Hotel's Website",
    "Cancellation policy improvement",
    "GURU bonus",
    "Commission Override",
    "Higher commission",
}

MEDIUM_EFFECT_SUBJECTS = {
    "Net rates improvement",
    "B2B rates improvement",
    "Opaque rate",
    "ECLC (Early Check-in / Late Check-out)",
    "New room categories",
    "Availability improvement",
    "Bedding correction",
    "Adding meals",
    "Package rates",
    "Top Position",
    "Top Stays",
    "GURU 1",
    "GURU 2",
    "GURU 3",
    "GURU 4",
    "Content issues",
    "B2B rates",
    "Net rates",
    "VCC",
    "Commercial meeting",
}

LONG_EFFECT_SUBJECTS = {
    "Hybrid",
    "Retention",
}

EFFECT_TYPE_LABELS = {
    "fast": "Быстрый",
    "medium": "Средний",
    "long": "Долгосрочный",
    "unknown": "Не определён",
}

EFFECT_MONTH_OFFSETS = {
    "fast": (-1, 0),
    "medium": (-1, 1),
    "long": (0, 2),
}

SUBJECT_EFFECT_MAP = {
    **{subject: "fast" for subject in FAST_EFFECT_SUBJECTS},
    **{subject: "medium" for subject in MEDIUM_EFFECT_SUBJECTS},
    **{subject: "long" for subject in LONG_EFFECT_SUBJECTS},
}

EXPECTED_SUBJECTS_COUNT = 34
