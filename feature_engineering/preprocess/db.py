from sqlalchemy import create_engine
from dbconfig import config


def create_db(config):
    """
    Creating database connection with postgresql server by configuration

    Param
    =========
    config: dictionary with necessary keys(user, passwd, host, port, db_name)

    Return
    =========
    Postgresql database engine
    """
    # setting up the db
    user = config['user'] 
    password = config['passwd'] 
    host = config['host'] 
    port = config['port'] 
    db_name = config['db_name']
    db_string = f"postgresql://{user}:{password}@{host}:{port}/{db_name}" 

    # create engine
    db = create_engine(db_string) 
    return db


db = create_db(config)
cursor = db.connect()