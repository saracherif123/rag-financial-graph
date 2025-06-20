import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'neo4j')

COMPANY_FILE = os.path.join('data', 'financial', 'company_info.json')

CREATE_COMPANY_QUERY = """
MERGE (c:Company {symbol: $symbol})
SET c.name = $name,
    c.sector = $sector,
    c.industry = $industry,
    c.market_cap = $market_cap,
    c.employees = $employees,
    c.description = $description,
    c.country = $country,
    c.website = $website
"""

def load_companies_to_neo4j(driver, companies):
    with driver.session(database=NEO4J_DATABASE) as session:
        for company in companies:
            session.write_transaction(
                lambda tx: tx.run(
                    CREATE_COMPANY_QUERY,
                    symbol=company.get('symbol', ''),
                    name=company.get('name', ''),
                    sector=company.get('sector', ''),
                    industry=company.get('industry', ''),
                    market_cap=company.get('market_cap', 0),
                    employees=company.get('employees', 0),
                    description=company.get('description', ''),
                    country=company.get('country', ''),
                    website=company.get('website', '')
                )
            )
            print(f"Loaded company: {company.get('symbol', '')}")

def main():
    if not (NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD):
        print("Please set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your .env file.")
        return
    if not os.path.exists(COMPANY_FILE):
        print(f"Company info file not found: {COMPANY_FILE}")
        return
    with open(COMPANY_FILE, 'r') as f:
        companies = json.load(f)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        load_companies_to_neo4j(driver, companies)
        print("All companies loaded successfully.")
    finally:
        driver.close()

if __name__ == "__main__":
    main() 