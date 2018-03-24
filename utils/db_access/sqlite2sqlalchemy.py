# encoding=utf-8
from datetime import datetime

from dataset import SNAPGowallaStockholm as Dataset
from sqlalchemy import Column, DateTime, ForeignKey, Integer, REAL, Table, UniqueConstraint
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.orm.exc import MultipleResultsFound, NoResultFound
from utils.db_access.db_access import DBManipulation

Base = declarative_base()

# Self-Referential Many-to-Many Relationship:
# http://docs.sqlalchemy.org/en/latest/orm/join_conditions.html#self-referential-many-to-many-relationship


edges = Table('edges',
              Base.metadata,
              Column('userid', Integer, ForeignKey('users.id'), primary_key=True),
              Column('friendid', Integer, ForeignKey('users.id'), primary_key=True))


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    global_userid = Column(Integer, unique=True)

    followers = relationship('User',
                             secondary='edges',
                             primaryjoin='User.id==edges.c.userid',
                             secondaryjoin='User.id==edges.c.friendid',
                             back_populates='followees')
    followees = relationship('User',
                             secondary='edges',
                             primaryjoin='User.id==edges.c.friendid',
                             secondaryjoin='User.id==edges.c.userid',
                             back_populates='followers')
    locations = relationship('Location',
                             secondary='checkins',
                             primaryjoin='User.id==checkins.c.userid',
                             secondaryjoin='checkins.c.locid==Location.id',
                             back_populates='visitors')
    checkins = relationship('Checkin',
                            back_populates='user')


class Location(Base):
    __tablename__ = 'locations'
    __table_args__ = (UniqueConstraint('lon', 'lat', name='gps'),)

    id = Column(Integer, primary_key=True)
    lon = Column(REAL)
    lat = Column(REAL)

    visitors = relationship('User',
                            secondary='checkins',
                            primaryjoin='Location.id==checkins.c.locid',
                            secondaryjoin='User.id==checkins.c.userid',
                            back_populates='locations')
    checkins = relationship('Checkin',
                            back_populates='location')


class Checkin(Base):
    __tablename__ = 'checkins'

    id = Column(Integer, primary_key=True)
    userid = Column(Integer, ForeignKey('users.id'))
    locdatetime = Column(DateTime)
    locid = Column(Integer, ForeignKey('locations.id'))

    user = relationship('User', back_populates='checkins')
    location = relationship('Location', back_populates='checkins')


dataset = Dataset
db = DBManipulation(dataset)

engine = create_engine(dataset.url)
Base.metadata.create_all(engine)
ScopedSession = scoped_session(sessionmaker(bind=engine))
session = ScopedSession()
batch_size = 10 ** 4


def get_locid(lon, lat):
    try:
        location = session.query(Location).filter(Location.lon == lon, Location.lat == lat).one()
    except MultipleResultsFound:
        pass
    except NoResultFound:
        pass
    else:
        return location.id


def get_userid(global_userid):
    try:
        user = session.query(User).filter(User.global_userid == global_userid).one()
    except MultipleResultsFound:
        pass
    except NoResultFound:
        pass
    else:
        return user.id


def str2datetime(string):
    return datetime.strptime(string, '%Y-%m-%d %H:%M:%S')


def import_users():
    db.cursor.execute(''.join(['SELECT DISTINCT userid',
                               ' FROM ', dataset.checkins,
                               ' ORDER BY userid ASC, locdatetime ASC']))
    batch_cnt = 0
    while True:
        batch_results = db.cursor.fetchmany(size=batch_size)
        if not batch_results:
            break
        batch_values = [{'id': idx + batch_cnt * batch_size, 'global_userid': row[0]}
                        for idx, row in enumerate(batch_results)]
        engine.execute(User.__table__.insert(), batch_values)
        batch_cnt += 1
        print(batch_cnt)


def import_locations():
    db.cursor.execute(''.join(['SELECT DISTINCT lon, lat',
                               ' FROM ', dataset.checkins,
                               ' ORDER BY userid ASC, locdatetime ASC']))
    batch_cnt = 0
    while True:
        batch_results = db.cursor.fetchmany(size=batch_size)
        if not batch_results:
            break
        batch_values = [{'id': idx + batch_cnt * batch_size, 'lon': row[0], 'lat': row[1]}
                        for idx, row in enumerate(batch_results)]
        engine.execute(Location.__table__.insert(), batch_values)
        batch_cnt += 1
        print(batch_cnt)


def import_checkins():
    db.cursor.execute(''.join(['SELECT userid, locdatetime, lon, lat',
                               ' FROM ', dataset.checkins,
                               ' ORDER BY userid ASC, locdatetime ASC']))
    batch_cnt = 0
    while True:
        batch_results = db.cursor.fetchmany(size=batch_size)
        if not batch_results:
            break
        batch_values = [{'id': idx + batch_cnt * batch_size, 'userid': get_userid(row[0]),
                         'time': str2datetime(row[1]), 'locid': get_locid(row[2], row[3])}
                        for idx, row in enumerate(batch_results)]
        engine.execute(Checkin.__table__.insert(), batch_values)
        batch_cnt += 1
        print(batch_cnt)


def import_edges():
    db.cursor.execute(''.join(['SELECT userid, friendid',
                               ' FROM ', dataset.edges]))
    batch_cnt = 0
    while True:
        batch_results = db.cursor.fetchmany(size=batch_size)
        if not batch_results:
            break
        batch_values = [{'userid': get_userid(global_userid), 'friendid': get_userid(global_friendid)}
                        for global_userid, global_friendid in batch_results]
        engine.execute(edges.insert(), batch_values)
        batch_cnt += 1
        print(batch_cnt)


def main():
    import_users()
    import_edges()
    import_locations()
    import_checkins()


if __name__ == '__main__':
    main()
