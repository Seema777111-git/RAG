from neo4j import GraphDatabase
import config

driver = GraphDatabase.driver(
    config.NEO4J_URI,
    auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
)

print(f"Testing URI: {config.NEO4J_URI}\n")

# 1. Try listing databases via system database
try:
    with driver.session(database="system") as session:
        result = session.run("SHOW DATABASES")
        print("Available Databases on this instance:")
        for r in result:
            print(f"  - Name: {r.get('name')} | Default: {r.get('default')} | Status: {r.get('currentStatus')}")
except Exception as e:
    print(f"Could not read from 'system' database: {e}")

# 2. Try default session without specifying any database
try:
    with driver.session() as session:
        val = session.run("RETURN 1 AS num").single()["num"]
        print(f"\nDefault session query succeeded (RETURN 1 = {val})")
except Exception as e:
    print(f"\nDefault session query failed: {e}")

driver.close()