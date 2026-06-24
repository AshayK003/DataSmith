Act as a security engineer.

Task:
Review this full-stack Python project for security issues and harden it using only free/open-source tools and libraries.

Check (in order of priority):
- authentication
- authorization
- input validation (all user-facing endpoints)
- output escaping
- secrets handling (API keys, env vars)
- CSRF
- SSRF
- XSS
- SQL injection (all DB queries)
- rate limiting
- file upload safety
- dependency vulnerabilities
- insecure defaults

The project is a Streamlit app at D:\Personal projects\DataSmith

Project structure:
- app.py - main entry point
- pages/01_Generate.py - main page (schema discovery, editing, dataset generation)
- pages/02_About.py - about page
- engine.py - core logic (schema resolution, knowledge graph, generation)
- knowledge_graph.py - domain/dataset knowledge graph
- generator.py - synthetic data generation logic
- crawler.py - web crawling for dataset schemas
- models.py - data models
- exporter.py - export functionality
- llm_client.py - LLM API client (used for schema discovery)
- config.py - configuration

Key areas of concern:
1. LLM client makes API calls with user-controlled prompts - check prompt injection guards
2. Knowledge graph stores crawled data - check data validation on ingest
3. Generator produces user-facing output - check for template injection
4. Config files and env vars - check secrets handling
5. All user input flows through Streamlit widgets - check validation

Do NOT hand-wave. Prioritize realistic attack paths based on the actual code.
Recommend fixes that fit the current Python/Streamlit stack. Avoid security theater.

Output format:
1. THREAT MODEL: What's the attack surface? Who are the threat actors?
2. VULNERABILITIES FOUND: Specific issues with file paths and line numbers
3. EXPLOITABILITY: For each vuln, how hard is it to exploit (easy/medium/hard)?
4. FIXES in priority order: Specific code changes for each issue
5. SECURE-BY-DEFAULT: What defaults should change to prevent future issues?

Do NOT modify any files. Just produce the analysis report.
