"""Text value profiles for realistic synthetic data generation.

Provides word banks and generators for common text column types
(location, merchant category, payment method, fraud labels, IDs, etc.)
instead of placeholder text like "Column Name 1".
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np

# ── Word banks ──────────────────────────────────────────────────────────────

CITIES_IN = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Chandigarh",
    "Indore", "Bhopal", "Nagpur", "Thane", "Visakhapatnam", "Patna",
    "Vadodara", "Guwahati", "Coimbatore", "Kochi", "Mysore", "Nashik",
    "Agra", "Varanasi", "Ranchi", "Bhubaneswar", "Amritsar", "Dehradun",
]

CITIES_WORLD = [
    "New York", "London", "Tokyo", "Dubai", "Singapore", "Sydney",
    "Paris", "Berlin", "Toronto", "San Francisco", "Shanghai", "Seoul",
]

MERCHANT_CATEGORIES = [
    "Grocery", "Restaurant", "Electronics", "Clothing", "Fuel",
    "Pharmacy", "Entertainment", "Utilities", "Travel", "Healthcare",
    "Ecommerce", "Education", "Insurance", "Telecom", "Transport",
    "Furniture", "Jewelry", "Sports", "Books", "Hardware",
]

PAYMENT_METHODS = [
    "Credit Card", "Debit Card", "UPI", "Net Banking", "Cash",
    "Wallet", "EMI", "NEFT", "RTGS", "Cheque",
]

FRAUD_LABELS = ["No", "Yes"]  # weighted ~95/5 below

FIRST_NAMES_IN = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
    "Ayaan", "Ishaan", "Shaurya", "Ananya", "Diya", "Myra", "Aanya",
    "Advika", "Prisha", "Ishita", "Navya", "Aaradhya", "Sara",
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Kumar", "Singh", "Gupta", "Reddy",
    "Joshi", "Nair", "Menon", "Iyer", "Deshmukh", "Das", "Choudhury",
    "Bose", "Sen", "Malhotra", "Kapoor", "Agarwal", "Mehta",
]

STATUSES = ["Active", "Inactive", "Pending", "Suspended", "Completed", "Failed"]
GENDERS = ["Male", "Female", "Other"]
RATING_LABELS = ["1", "2", "3", "4", "5"]
COUNTRIES = ["India", "USA", "UK", "Canada", "Australia", "Germany", "Japan", "UAE"]
CATEGORIES_ABC = ["A", "B", "C", "D"]
BOOLEAN_YESNO = ["Yes", "No"]
EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
                 "rediffmail.com", "proton.me"]

# ── Helpers ─────────────────────────────────────────────────────────────────


def _id_generator(prefix: str, digits: int = 5) -> Callable:
    """Return a function that generates IDs like PREFIX-00001."""
    fmt = f"{prefix}-{{:0{digits}d}}"

    def _gen(n: int, rng: np.random.Generator, **_) -> np.ndarray:
        # Random offsets so IDs aren't sequential 1,2,3
        ids = rng.integers(1, 10**digits, size=n)
        return np.array([fmt.format(i) for i in ids])

    return _gen


def _categorical_list(items: list[str], weights: list[float] | None = None
                       ) -> Callable:
    """Return a generator that picks randomly from a word list."""
    def _gen(n: int, rng: np.random.Generator, **_) -> np.ndarray:
        return np.array(rng.choice(items, size=n, p=weights))
    return _gen


def _template_from_desc(desc: str) -> Callable:
    """Return a generator that creates values from a description pattern."""
    # Extract keywords from the description to build a minimal template
    words = re.findall(r"[\w']+", desc.lower())
    # Use first few notable content words as a base
    content_words = [w for w in words if len(w) > 3 and w not in
                     {"that", "this", "with", "from", "where", "which",
                      "their", "about", "would", "could", "should", "what",
                      "when", "for", "the", "and", "are", "was", "has", "had"}]
    if content_words:
        base = content_words[0].title()
        # Generate variations: e.g. "Value_1", "Value_2"
        def _gen(n: int, rng: np.random.Generator, **_) -> np.ndarray:
            suffixes = rng.integers(1, 1000, size=n)
            return np.array([f"{base}_{s}" for s in suffixes])
        return _gen
    return _categorical_list(["Sample A", "Sample B", "Sample C"])


# ── Pattern matcher ─────────────────────────────────────────────────────────

# Match rules: (regex_pattern, generator_factory_or_list)
# More specific patterns should come first.
_TEXT_RULES: list[tuple[re.Pattern, Callable | list]] = [
    # IDs — proper formatted IDs
    (re.compile(r"(transaction|trn|txn|order|invoice)_?(id|num|number|ref|code)", re.I),
     _id_generator("TRN", 6)),
    (re.compile(r"(customer|user|client|member|patient)_?(id|num|number)", re.I),
     _id_generator("CUST", 5)),
    (re.compile(r"(product|item)_?(id|num|code|sku)", re.I),
     _id_generator("PRD", 4)),
    (re.compile(r"(employee|emp|staff)_?(id|num)", re.I),
     _id_generator("EMP", 4)),
    (re.compile(r"(order|purchase)_?(id|num)", re.I),
     _id_generator("ORD", 6)),
    (re.compile(r"(account)_?(id|num)", re.I),
     _id_generator("ACC", 6)),

    # Location-like columns
    (re.compile(r"^(city|town|location|place|region|district)$", re.I),
     _categorical_list(CITIES_IN)),
    (re.compile(r"(city|town|location|place)", re.I),
     _categorical_list(CITIES_IN)),
    (re.compile(r"country", re.I),
     _categorical_list(COUNTRIES)),
    (re.compile(r"address", re.I),
     _categorical_list(CITIES_IN)),  # fallback, could be richer

    # Categories
    (re.compile(r"(merchant|product|item)_?(categ|type|class|kind)", re.I),
     _categorical_list(MERCHANT_CATEGORIES)),
    (re.compile(r"(categ|type|class|kind|segment)", re.I),
     _categorical_list(CATEGORIES_ABC)),
    (re.compile(r"department", re.I),
     _categorical_list(["Engineering", "Sales", "Marketing", "Finance",
                        "HR", "Operations", "Support"])),
    (re.compile(r"payment.*(method|type|mode)", re.I),
     _categorical_list(PAYMENT_METHODS)),

    # Labels & status
    (re.compile(r"(fraud|is_fraud|fraudulent)", re.I),
     _categorical_list(FRAUD_LABELS, weights=[0.95, 0.05])),
    (re.compile(r"(status|state|condition)", re.I),
     _categorical_list(STATUSES)),
    (re.compile(r"^(gender|sex)$", re.I),
     _categorical_list(GENDERS)),
    (re.compile(r"^(rating|score|grade|rank)$", re.I),
     _categorical_list(RATING_LABELS)),
    (re.compile(r"^(yes_no|is_active|active|flag)$", re.I),
     _categorical_list(BOOLEAN_YESNO)),

    # Names
    (re.compile(r"^(name|full_name|customer_name|user_name)$", re.I),
     lambda n, rng, **_: np.array([
         f"{rng.choice(FIRST_NAMES_IN)} {rng.choice(LAST_NAMES)}"
         for _ in range(n)
     ])),
    (re.compile(r"^(first_name|fname)$", re.I),
     _categorical_list(FIRST_NAMES_IN)),
    (re.compile(r"^(last_name|lname|surname)$", re.I),
     _categorical_list(LAST_NAMES)),

    # Contact
    (re.compile(r"email", re.I),
     lambda n, rng, **_: np.array([
         f"{rng.choice(FIRST_NAMES_IN).lower()}.{rng.choice(LAST_NAMES).lower()}"
         f"@{rng.choice(EMAIL_DOMAINS)}"
         for _ in range(n)
     ])),
    (re.compile(r"(phone|mobile|contact|cell)", re.I),
     lambda n, rng, **_: np.array([
         f"+91-{rng.integers(70000, 99999, 1)[0]}{rng.integers(10000, 99999, 1)[0]}"
         for _ in range(n)
     ])),

    # Domains / URLs
    (re.compile(r"^(domain|website|url|site)$", re.I),
     lambda n, rng, **_: np.array([
         f"{rng.choice(FIRST_NAMES_IN).lower()}.com" for _ in range(n)
     ])),
]


def choose_text_generator(col_name: str, description: str = ""
                          ) -> Callable | None:
    """Find the best text generator for a column based on name and description.

    Returns a callable with signature ``gen(n, rng, **params) -> np.ndarray``,
    or ``None`` if no rule matches (caller falls back to placeholder).
    """
    name_lower = col_name.lower().replace("_", " ").replace("-", " ").strip()

    for pattern, factory in _TEXT_RULES:
        if pattern.search(name_lower) or (description and pattern.search(description)):
            if callable(factory):
                return factory
            if isinstance(factory, list):
                return _categorical_list(factory)
    return None
