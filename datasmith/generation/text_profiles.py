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

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Kolkata",
    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Chandigarh",
    "Indore", "Bhopal", "Nagpur", "Thane", "Visakhapatnam", "Patna",
    "Vadodara", "Guwahati", "Coimbatore", "Kochi", "Mysore", "Nashik",
    "Agra", "Varanasi", "Ranchi", "Bhubaneswar", "Amritsar", "Dehradun",
    "New York", "London", "Tokyo", "Dubai", "Singapore", "Sydney",
    "Paris", "Berlin", "Toronto", "San Francisco", "Shanghai", "Seoul",
    "Rome", "Madrid", "Amsterdam", "Bangkok", "Vienna", "Prague",
    "Oslo", "Stockholm", "Copenhagen", "Helsinki", "Zurich", "Munich",
    "Milan", "Barcelona", "Lisbon", "Dublin", "Edinburgh", "Athens",
    "Istanbul", "Moscow", "Beijing", "Hong Kong", "Kuala Lumpur", "Jakarta",
    "Manila", "Ho Chi Minh City", "Lahore", "Dhaka", "Cairo",
    "Casablanca", "Nairobi", "Cape Town", "Lagos", "Sao Paulo",
    "Buenos Aires", "Lima", "Bogota", "Santiago", "Mexico City",
    "Los Angeles", "Chicago", "Houston", "Seattle", "Boston",
    "Warsaw", "Budapest", "Bucharest", "Kiev", "Bratislava",
    "Reykjavik", "Tallinn", "Vilnius", "Riga", "Ljubljana",
    "Montevideo", "Panama City", "San Jose", "San Juan", "Quito",
    "Auckland", "Wellington", "Perth", "Melbourne", "Brisbane",
    "Hamburg", "Frankfurt", "Stuttgart", "Dusseldorf", "Cologne",
    "Lyon", "Marseille", "Nice", "Toulouse", "Bordeaux",
]

MERCHANT_CATEGORIES = [
    "Grocery", "Restaurant", "Electronics", "Clothing", "Fuel",
    "Pharmacy", "Entertainment", "Utilities", "Travel", "Healthcare",
    "Ecommerce", "Education", "Insurance", "Telecom", "Transport",
    "Furniture", "Jewelry", "Sports", "Books", "Hardware",
    "Automotive", "Beauty", "Home Improvement", "Pet Supplies", "Office Supplies",
    "Agriculture", "Legal Services", "Consulting", "Real Estate", "Logistics",
]

PAYMENT_METHODS = [
    "Credit Card", "Debit Card", "UPI", "Net Banking", "Cash",
    "Wallet", "EMI", "NEFT", "RTGS", "Cheque",
    "Google Pay", "Apple Pay", "PayPal", "Cryptocurrency", "Bank Transfer",
]

FRAUD_LABELS = ["No", "Yes"]

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
    "Ayaan", "Ishaan", "Shaurya", "Ananya", "Diya", "Myra", "Aanya",
    "Advika", "Prisha", "Ishita", "Navya", "Aaradhya", "Sara",
    "Oliver", "Emma", "Liam", "Sophia", "Noah", "Isabella", "Ethan",
    "Mia", "Lucas", "Charlotte", "James", "Amelia", "Benjamin",
    "Harper", "Elijah", "Evelyn", "William", "Abigail", "Henry",
    "Emily", "Alexander", "Ella", "Daniel", "Avery", "Michael",
    "Scarlett", "Sebastian", "Grace", "Jack", "Chloe", "Owen",
    "Victoria", "Samuel", "Riley", "David", "Aria", "Joseph",
    "Lily", "John", "Zoey", "Leo", "Penelope", "Gabriel",
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Kumar", "Singh", "Gupta", "Reddy",
    "Joshi", "Nair", "Menon", "Iyer", "Deshmukh", "Das", "Choudhury",
    "Bose", "Sen", "Malhotra", "Kapoor", "Agarwal", "Mehta",
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia",
    "Miller", "Davis", "Rodriguez", "Martinez", "Wilson", "Anderson",
    "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee",
    "Thompson", "White", "Harris", "Clark", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott",
    "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams",
    "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner",
]

STATUSES = ["Active", "Inactive", "Pending", "Suspended", "Completed", "Failed",
            "Approved", "Rejected", "Processing", "Cancelled", "On Hold", "Delivered"]

GENDERS = ["Male", "Female", "Other"]
RATING_LABELS = ["1", "2", "3", "4", "5"]
BOOLEAN_YESNO = ["Yes", "No"]

COUNTRIES = [
    "India", "USA", "UK", "Canada", "Australia", "Germany", "Japan", "UAE",
    "Brazil", "Mexico", "France", "Italy", "Spain", "Netherlands", "Sweden",
    "Norway", "Denmark", "Finland", "Switzerland", "Austria", "Belgium",
    "Ireland", "Portugal", "Greece", "Poland", "Czech Republic", "Hungary",
    "Romania", "Russia", "Turkey", "Israel", "Saudi Arabia", "Qatar",
    "Kuwait", "Oman", "Bahrain", "Jordan", "Egypt", "South Africa", "Nigeria",
    "Kenya", "Ghana", "Morocco", "Argentina", "Chile", "Colombia", "Peru",
    "Thailand", "Vietnam", "Indonesia", "Malaysia", "Philippines", "Singapore",
    "South Korea", "Taiwan", "China", "New Zealand", "Pakistan", "Bangladesh",
    "Sri Lanka", "Nepal", "Myanmar", "Ukraine", "Kazakhstan", "Algeria",
    "Angola", "Ethiopia", "Tanzania", "Uganda", "Mozambique", "Zambia",
    "Zimbabwe", "Botswana", "Ivory Coast", "Cameroon", "Tunisia", "Sudan",
    "Senegal", "Mali", "Madagascar", "Burkina Faso", "Benin", "Rwanda",
    "Somalia", "Afghanistan", "Uzbekistan", "Azerbaijan", "Georgia",
    "Armenia", "Belarus", "Croatia", "Serbia", "Bulgaria", "Slovakia",
    "Slovenia", "Lithuania", "Latvia", "Estonia", "Costa Rica", "Panama",
    "Guatemala", "Dominican Republic", "Puerto Rico", "Uruguay", "Paraguay",
    "Bolivia", "Ecuador", "Venezuela", "Cuba", "Jamaica", "Trinidad",
    "Bahamas", "Barbados", "Fiji", "Papua New Guinea", "Mongolia", "Laos",
    "Cambodia", "Lebanon", "Syria", "Libya", "Iraq", "Yemen", "Maldives",
    "Bhutan", "Brunei", "Macau", "Hong Kong", "Luxembourg", "Monaco",
    "Malta", "Cyprus", "Iceland", "Montenegro", "Albania", "North Macedonia",
    "Bosnia", "Moldova", "Kyrgyzstan", "Turkmenistan", "Tajikistan",
    "Mauritius", "Seychelles", "Comoros", "Cape Verde", "Mauritania",
    "Chad", "Niger", "Congo", "DRC", "Gabon", "Equatorial Guinea",
    "Liberia", "Sierra Leone", "Togo", "Eritrea", "Djibouti", "Burundi",
    "Lesotho", "Eswatini", "Timor-Leste", "Solomon Islands", "Vanuatu",
    "Samoa", "Tonga", "Palau", "Micronesia", "Marshall Islands",
    "Antarctica", "Greenland", "New Caledonia", "French Polynesia", "Bermuda",
    "Cayman Islands", "Aruba", "Monaco", "Liechtenstein", "San Marino",
]

CATEGORIES_ABC = ["A", "B", "C", "D", "E", "F"]

BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

DIAGNOSES = [
    "Type 2 Diabetes", "Hypertension", "Asthma", "COVID-19", "Pneumonia",
    "Acute Bronchitis", "Urinary Tract Infection", "Fractured Tibia",
    "Migraine", "Anxiety Disorder", "Major Depressive Disorder",
    "Osteoarthritis", "Lower Back Pain", "Gastroenteritis",
    "Cellulitis", "Dehydration", "Anemia", "Hyperthyroidism",
    "Chronic Kidney Disease", "Coronary Artery Disease",
]

MAJORS = [
    "Computer Science", "Business Administration", "Biology",
    "Mechanical Engineering", "Psychology", "Economics",
    "Electrical Engineering", "Political Science", "English Literature",
    "Nursing", "Accounting", "Marketing", "Chemistry",
    "Civil Engineering", "History", "Mathematics", "Physics",
    "Philosophy", "Communications", "Environmental Science",
]

PRIORITY_LEVELS = ["Critical", "High", "Medium", "Low"]

HOTEL_NAMES = [
    "Grand Plaza", "Seaside Resort", "Mountain Lodge", "City Inn",
    "Royal Suites", "Harbor View", "Sunset Hotel", "Skyline Tower",
    "Garden Retreat", "Crystal Hotel", "Ocean Breeze", "Elite Stay",
]

EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com",
                 "rediffmail.com", "proton.me", "icloud.com", "aol.com",
                 "zoho.com", "mail.com", "fastmail.com", "gmx.com"]

COMPANIES = [
    "Acme Corp", "TechVista", "DataFlow Inc", "CloudPeak Systems",
    "NovaWorks", "BrightPath Solutions", "Quantum Software", "Pinnacle Group",
    "Summit Analytics", "Vertex Technologies", "Horizon Labs", "Meridian Corp",
    "Atlas Innovations", "Catalyst Partners", "Ridgeway Industries",
    "Peak Performance Inc", "NorthStar Consulting", "Titan Machinery",
    "Evergreen Solutions", "Pacific Rim Trading", "Sapphire Systems",
    "Cobalt Technologies", "Apex Digital", "Orion Networks", "Vanguard Corp",
]

JOB_TITLES = [
    "Software Engineer", "Data Analyst", "Product Manager", "Sales Executive",
    "Marketing Manager", "Operations Lead", "Financial Analyst", "HR Specialist",
    "Customer Success Manager", "Business Analyst", "DevOps Engineer",
    "UX Designer", "Content Writer", "Account Executive", "Project Manager",
    "Research Scientist", "Accountant", "Legal Counsel", "Support Engineer",
    "Solutions Architect", "Data Scientist", "Engineering Manager",
    "Technical Writer", "QA Engineer", "Product Designer",
]

DEPARTMENTS = [
    "Engineering", "Sales", "Marketing", "Finance", "HR",
    "Operations", "Support", "Legal", "Product", "Research",
    "Design", "Data", "Security", "Infrastructure", "Analytics",
]

PRODUCTS = [
    "CloudSync Pro", "DataPulse", "SecureVault", "FlowManager",
    "InsightBoard", "ConnectHub", "TaskForce", "Analytix",
    "CyberGuard", "SmartOps", "Velocity CRM", "PixelStudio",
    "TalentScout", "InvoiceFlow", "CampaignPro",
]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _id_generator(prefix: str, digits: int = 5) -> Callable:
    """Return a function that generates IDs like PREFIX-00001."""
    fmt = f"{prefix}-{{:0{digits}d}}"

    def _gen(n: int, rng: np.random.Generator, **_) -> np.ndarray:
        ids = rng.integers(1, 10**digits, size=n)
        return np.array([fmt.format(i) for i in ids])

    return _gen


def _categorical_list(
    items: list[str], weights: list[float] | None = None
) -> Callable:
    """Return a generator that picks randomly from a word list."""
    def _gen(n: int, rng: np.random.Generator, **_) -> np.ndarray:
        return np.array(rng.choice(items, size=n, p=weights))
    return _gen


def _sentence_generator() -> Callable:
    """Return a generator that produces varied descriptive text.

    Used as a catch-all for unknown text columns to avoid the
    "Placeholder 1, Placeholder 2" look. Generates realistic-sounding
    short sentences from random template combinations.
    """
    subjects = [
        "The system", "This process", "The request", "The application",
        "Our platform", "The service", "The operation", "This task",
        "The transaction", "The record", "The entry", "This account",
    ]
    verbs = [
        "requires", "processes", "handles", "manages", "generates",
        "produces", "supports", "triggers", "initiates", "completes",
        "validates", "confirms", "updates", "maintains", "schedules",
    ]
    objects = [
        "high-priority orders", "customer requests", "data records",
        "payment transactions", "system updates", "user profiles",
        "inventory items", "service tickets", "compliance checks",
        "quality reviews", "batch operations", "scheduled tasks",
        "automated workflows", "processing pipelines", "access requests",
    ]
    qualifiers = [
        "via API", "on demand", "in real-time", "per schedule",
        "through web interface", "automatically", "manually",
        "with approval", "in background", "via batch",
    ]

    # Pre-compose 180 unique patterns
    _patterns = [
        f"{s} {v} {o} {q}."
        for s in subjects
        for v in verbs[:5]
        for o in objects[:3]
        for q in qualifiers[:3]
    ][:180]  # Keep it manageable

    def _gen(n: int, rng: np.random.Generator, **_) -> np.ndarray:
        picks = rng.integers(0, len(_patterns), size=n)
        return np.array([_patterns[int(i)] for i in picks])

    return _gen


_gen_sentence = _sentence_generator()


def _template_from_desc(desc: str) -> Callable:
    """Return a generator that creates values from a description pattern."""
    # Extract keywords from the description to build a minimal template
    words = re.findall(r"[\w']+", desc.lower())
    content_words = [w for w in words if len(w) > 3 and w not in
                     {"that", "this", "with", "from", "where", "which",
                      "their", "about", "would", "could", "should", "what",
                      "when", "for", "the", "and", "are", "was", "has", "had"}]
    if content_words:
        base = content_words[0].title()

        def _gen(n: int, rng: np.random.Generator, **_) -> np.ndarray:
            suffixes = rng.integers(1, 10000, size=n)
            return np.array([f"{base}_{s}" for s in suffixes])
        return _gen
    return _categorical_list(["Sample A", "Sample B", "Sample C"])


# ── Pattern matcher ─────────────────────────────────────────────────────────

_TEXT_RULES: list[tuple[re.Pattern, Callable | list]] = [
    # IDs — proper formatted IDs
    (re.compile(r"(transaction|trn|txn|order|invoice)[ _-]?(id|num|number|ref|code)", re.I),
     _id_generator("TRN", 6)),
    (re.compile(r"(customer|user|client|member|patient)[ _-]?(id|num|number)", re.I),
     _id_generator("CUST", 5)),
    (re.compile(r"(product|item)[ _-]?(id|num|code|sku)", re.I),
     _id_generator("PRD", 4)),
    (re.compile(r"(employee|emp|staff)[ _-]?(id|num)", re.I),
     _id_generator("EMP", 4)),
    (re.compile(r"(order|purchase)[ _-]?(id|num)", re.I),
     _id_generator("ORD", 6)),
    (re.compile(r"(account)[ _-]?(id|num)", re.I),
     _id_generator("ACC", 6)),
    (re.compile(r"(policy|claim)[ _-]?(id|num|number)", re.I),
     _id_generator("POL", 7)),
    (re.compile(r"(booking|reservation)[ _-]?(id|num|number|ref|code)", re.I),
     _id_generator("BKG", 6)),
    (re.compile(r"(ticket|incident|case)[ _-]?(id|num|number|ref)", re.I),
     _id_generator("TCK", 6)),
    (re.compile(r"(student|enrol|registration)[ _-]?(id|num|number)", re.I),
     _id_generator("STU", 6)),

    # Location-like columns
    (re.compile(r"^(city|town|location|place|region|district)$", re.I),
     _categorical_list(CITIES)),
    (re.compile(r"(city|town|location|place)", re.I),
     _categorical_list(CITIES)),
    (re.compile(r"country", re.I),
     _categorical_list(COUNTRIES)),


    # Contact
    (re.compile(r"email", re.I),
     lambda n, rng, **_: np.array([
         f"{rng.choice(FIRST_NAMES).lower()}.{rng.choice(LAST_NAMES).lower()}"
         f"@{rng.choice(EMAIL_DOMAINS)}"
         for _ in range(n)
     ])),
    (re.compile(r"address", re.I),
     _categorical_list(CITIES)),

    # Organization names
    (re.compile(r"(company|organization|vendor|supplier|partner)", re.I),
     _categorical_list(COMPANIES)),
    (re.compile(r"(department|team|division|unit)", re.I),
     _categorical_list(DEPARTMENTS)),
    (re.compile(r"(job[ _-]?title|position|designation|role)", re.I),
     _categorical_list(JOB_TITLES)),
    (re.compile(r"(product[ _-]?name|item[ _-]?name|service[ _-]?name)", re.I),
     _categorical_list(PRODUCTS)),
    (re.compile(r"(hotel[ _-]?name|hotel[ _-]?brand|lodging[ _-]?name)", re.I),
     _categorical_list(HOTEL_NAMES)),

    # Categories
    (re.compile(r"(merchant|product|item)[ _-]?(categ|type|class|kind)", re.I),
     _categorical_list(MERCHANT_CATEGORIES)),
    (re.compile(r"blood[ _-]?type", re.I),
     _categorical_list(BLOOD_TYPES)),
    (re.compile(r"(categ|type|class|kind|segment)", re.I),
     _categorical_list(CATEGORIES_ABC)),
    (re.compile(r"payment.*(method|type|mode)", re.I),
     _categorical_list(PAYMENT_METHODS)),

    # Labels & status
    (re.compile(r"(fraud|is_fraud|fraudulent)", re.I),
     _categorical_list(FRAUD_LABELS, weights=[0.95, 0.05])),
    (re.compile(r"(status|state|condition)", re.I),
     _categorical_list(STATUSES)),
    (re.compile(r"(priority|urgency|severity|escalation[ _-]?level)", re.I),
     _categorical_list(PRIORITY_LEVELS)),
    (re.compile(r"^(gender|sex)$", re.I),
     _categorical_list(GENDERS)),
    (re.compile(r"^(rating|score|grade|rank)$", re.I),
     _categorical_list(RATING_LABELS)),
    (re.compile(r"^(yes_no|is_active|active|flag)$", re.I),
     _categorical_list(BOOLEAN_YESNO)),

    # Names
    (re.compile(
        r"^(name|full_name|customer_name|user_name|assigned_to|reported_by|contact_person)$",
        re.I),
     lambda n, rng, **_: np.array([
         f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
         for _ in range(n)
     ])),
    (re.compile(r"^(first_name|fname)$", re.I),
     _categorical_list(FIRST_NAMES)),
    (re.compile(r"^(last_name|lname|surname)$", re.I),
     _categorical_list(LAST_NAMES)),
    (re.compile(r"(phone|mobile|contact|cell)", re.I),
     lambda n, rng, **_: np.array([
         f"+91-{rng.integers(70000, 99999, 1)[0]}{rng.integers(10000, 99999, 1)[0]}"
         for _ in range(n)
     ])),

    # Domains / URLs
    (re.compile(r"^(domain|website|url|site)$", re.I),
     lambda n, rng, **_: np.array([
         f"{rng.choice(FIRST_NAMES).lower()}.com" for _ in range(n)
     ])),

    # Medical / diagnoses
    (re.compile(r"(diagnosis|diagnoses|medical[ _-]?condition|ailment)", re.I),
     _categorical_list(DIAGNOSES)),
    (re.compile(r"(major|field[ _-]?of[ _-]?study|discipline|specialization|concentration)", re.I),
     _categorical_list(MAJORS)),

    # Description / notes columns
    (re.compile(r"(description|desc|summary|notes?|comment|feedback|review|remark)", re.I),
     _gen_sentence),

    # Catch-all — generate varied sentences instead of "Field Value 1"
    (re.compile(r".*"), _gen_sentence),
]


def choose_text_generator(col_name: str, description: str = ""
                          ) -> Callable | None:
    """Find the best text generator for a column based on name and description.

    Returns a callable with signature ``gen(n, rng, **params) -> np.ndarray``.
    The catch-all ``.*`` rule at the end ensures every text column gets a
    generator — this function returns **None only** if something goes wrong.

    Matching is two-pass: column name first (exact), then description
    (broader). This prevents description keywords like ``address`` from
    hijacking columns like ``email`` whose name would match a later rule.
    """
    name_lower = col_name.lower().replace("-", " ").strip()

    # Pass 1: match column name only (highest priority)
    for pattern, factory in _TEXT_RULES:
        if pattern.search(name_lower):
            if callable(factory):
                return factory
            if isinstance(factory, list):
                return _categorical_list(factory)

    # Pass 2: match description only (fallback, avoids rule-order pitfalls)
    if description:
        desc_lower = description.lower()
        for pattern, factory in _TEXT_RULES:
            if pattern.search(desc_lower):
                if callable(factory):
                    return factory
                if isinstance(factory, list):
                    return _categorical_list(factory)

    # Should never reach here due to catch-all .* rule, but just in case
    return None
