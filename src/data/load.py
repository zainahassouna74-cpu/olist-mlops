import pandas as pd
from sqlalchemy import create_engine, inspect

from src.utils.config import load_config


def create_database_engine():
    config = load_config()
    db = config["database"]

    connection_string = (
        f"postgresql+psycopg2://{db['user']}:postgres"
        f"@{db['host']}:{db['port']}/{db['database']}"
    )

    return create_engine(connection_string)


def load_database_tables(engine=None):
    if engine is None:
        engine = create_database_engine()

    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    tables = {}

    for table_name in table_names:
        tables[table_name] = pd.read_sql(
            f'SELECT * FROM "{table_name}"',
            engine,
        )

    return tables
