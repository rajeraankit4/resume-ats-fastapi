DOMAINS = {
    "Software": "SDE, QA, DevOps, Cybersecurity, and software/platform roles",
    "Analyst": "Data Analyst, Business Analyst, ML/AI, and analytics roles",
    "Core_NonTech": "Core/NonTech roles",
}

DOMAIN_PROTOTYPES = {
    "Software": (
        "Software engineering backend frontend full stack QA test automation DevOps cloud "
        "infrastructure cybersecurity application development APIs web development databases."
    ),
    "Analyst": (
        "Data analyst business analyst research analyst data science machine learning "
        "artificial intelligence reporting dashboards business intelligence forecasting "
        "statistics data visualization experimentation."
    ),
    "Core_NonTech": (
        "Core engineering embedded systems firmware hardware electronics electrical mechanical "
        "manufacturing automotive operations procurement supply chain sales marketing HR finance "
        "administration non-technical roles."
    ),
}

DOMAIN_SKILL_TERMS = {
    "Software": {
        "software", "developer", "backend", "frontend", "full stack", "web", "api", "rest api",
        "graphql", "node.js", "react", "angular", "vue", "express", "javascript", "typescript",
        "html", "css", "java", "python", "c++", "sql", "mongodb", "postgresql", "redis",
        "git", "github", "docker", "aws", "gcp", "azure", "linux", "kubernetes", "terraform",
        "jenkins", "github actions", "devops", "cloud", "microservices", "system design",
        "qa", "selenium", "cypress", "jest", "regression testing", "security", "cybersecurity",
        "owasp", "siem", "soc", "network security", "vulnerability assessment",
        "computer science", "information technology", "cs / it", "programming",
        "data structures", "algorithms", "object-oriented", "oops",
    },
    "Analyst": {
        "data analyst", "business analyst", "research analyst", "data science", "machine learning",
        "artificial intelligence", "deep learning", "nlp", "computer vision", "python", "sql",
        "excel", "power bi", "tableau", "looker", "reporting", "dashboard", "kpi",
        "business intelligence", "forecasting", "statistics", "data visualization",
        "scikit-learn", "tensorflow", "pytorch", "pandas", "numpy", "spark", "etl",
        "feature engineering", "llm", "transformer", "market research", "requirements gathering",
        "stakeholder management", "process improvement", "financial modelling",
    },
    "Core_NonTech": {
        "embedded systems", "firmware", "hardware", "electronics", "electrical", "mechanical",
        "manufacturing", "automotive", "tractor", "industrial machinery", "microcontroller",
        "rtos", "pcb", "vhdl", "verilog", "uart", "spi", "i2c", "iot", "field service",
        "quality control", "operations", "supply chain", "procurement", "sales", "marketing",
        "human resources", "finance", "accounting", "administration", "business development",
        "recruitment", "customer service", "admission counsellor", "logistics", "retail",
        "civil", "maintenance", "utility", "transformers", "dg sets", "hvac",
    },
}

DOMAIN_INTENT_PATTERNS = {
    "Software": [
        r"\bsoftware engineer",
        r"\bsoftware development\b",
        r"\bbackend\b",
        r"\bfrontend\b",
        r"\bfull stack\b",
        r"\bweb developer\b",
        r"\bapplication development\b",
        r"\bdevops\b",
        r"\bcybersecurity\b",
        r"\bqa\b",
    ],
    "Analyst": [
        r"\bdata science\b",
        r"\banalytics\b",
        r"\bdata analyst\b",
        r"\bbusiness analyst\b",
        r"\bmachine learning\b",
        r"\bartificial intelligence\b",
        r"\bpower bi\b",
        r"\bdashboard\b",
        r"\beda\b",
        r"\bdata analysis\b",
        r"\bdata-driven\b",
        r"\bsql queries?\b",
    ],
    "Core_NonTech": [
        r"\bembedded\b",
        r"\bfirmware\b",
        r"\belectrical\b",
        r"\bmechanical\b",
        r"\bmanufacturing\b",
        r"\bhardware\b",
        r"\boperations\b",
    ],
}

NON_JD_PATTERNS = [
    r"\bapplication guideline",
    r"\bsteps to apply\b",
    r"\bcareer portal\b",
    r"\bmanage your profile\b",
    r"\bcreate an account\b",
    r"\be-signing\b",
    r"\bcandidate email\b",
    r"\bfield specific instructions\b",
    r"\bnavigation and guidelines\b",
    r"\bbrowse the open role\b",
    r"\bhiring process\b",
]

JD_PATTERNS = [
    r"\bjob description\b",
    r"\bposition summary\b",
    r"\brole summary\b",
    r"\bresponsibilit",
    r"\bqualification",
    r"\brequirements?\b",
    r"\btechnical skills?\b",
    r"\bpreferred skills?\b",
    r"\bexperience\b",
    r"\bwhat you'll do\b",
    r"\babout the role\b",
]

SEMANTIC_RELEVANCE_GATE = 0.25
WEIGHT_SEMANTIC = 0.50
WEIGHT_SKILL = 0.30
WEIGHT_KEYWORD = 0.20
DOMAIN_FIT_WEIGHT = 0.65
DOMAIN_BUCKET_WEIGHT = 0.35

PRIORITY_PATTERNS = [
    r"\brequired\b",
    r"\bpreferred\b",
    r"\bqualification",
    r"\bresponsibilit",
    r"\bjob description\b",
    r"\brole summary\b",
    r"\btechnical skills?\b",
    r"\bprogramming\b",
    r"\bsoftware\b",
    r"\bdeveloper\b",
    r"\bengineer\b",
    r"\bbackend\b",
    r"\bfrontend\b",
    r"\bfull[\s-]?stack\b",
    r"\bapi\b",
    r"\bdatabase\b",
    r"\bpython\b",
    r"\bjava(script)?\b",
    r"\breact\b",
    r"\bnode(\.js)?\b",
    r"\bc\+\+\b",
    r"\bsql\b",
    r"\baws\b",
    r"\bgit\b",
    r"\bdata\b",
    r"\bmachine learning\b",
    r"\bcloud\b",
    r"\bdevops\b",
    r"\bkubernetes\b",
    r"\bdocker\b",
    r"\bsecurity\b",
    r"\bembedded\b",
    r"\bfirmware\b",
]

RELEVANCE_QUERIES = [
    "Required technical skills, programming languages, frameworks, tools, and education.",
    "Core job responsibilities, engineering tasks, deliverables, and day to day work.",
    "Candidate qualifications, must have requirements, preferred experience, and role expectations.",
]

