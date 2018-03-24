# encoding=utf-8
import numpy as np
from dataset import GowallaSQLAlchemy
from dateutil.parser import parse
from pandas import read_csv
from sqlalchemy import Column, create_engine, Integer, DateTime, ForeignKey, REAL, Table, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, scoped_session, sessionmaker

Base = declarative_base()

edges = Table('edges',
              Base.metadata,
              Column('userid', Integer, ForeignKey('users.id'), primary_key=True),
              Column('friendid', Integer, ForeignKey('users.id'), primary_key=True))


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    bookmarked_spots_count = Column(Integer)
    challenge_pin_count = Column(Integer)
    checkin_num = Column(Integer)
    country_pin_count = Column(Integer)
    friends_count = Column(Integer)
    global_userid = Column(Integer, unique=True)
    highlights_count = Column(Integer)
    items_count = Column(Integer)
    photos_count = Column(Integer)
    pins_count = Column(Integer)
    places_num = Column(Integer)
    province_pin_count = Column(Integer)
    region_pin_count = Column(Integer)
    stamps_count = Column(Integer)
    state_pin_count = Column(Integer)
    trips_count = Column(Integer)

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
    # __table_args__ = (UniqueConstraint('lon', 'lat', name='gps'),)

    id = Column(Integer, primary_key=True)
    lon = Column(REAL)
    lat = Column(REAL)
    name = Column(Text)
    city_state = Column(Text)
    created_at = Column(DateTime)
    photos_count = Column(Integer)
    checkins_count = Column(Integer)
    users_count = Column(Integer)
    radius_meters = Column(Integer)
    highlights_count = Column(Integer)
    items_count = Column(Integer)
    max_items_count = Column(Integer)
    spot_categories = Column(Text)

    visitors = relationship('User',
                            secondary='checkins',
                            primaryjoin='Location.id==checkins.c.locid',
                            secondaryjoin='User.id==checkins.c.userid',
                            back_populates='locations')
    checkins = relationship('Checkin',
                            back_populates='location')


class Checkin(Base):
    __tablename__ = 'checkins'

    id = Column(Integer, primary_key=True, autoincrement=True)
    userid = Column(Integer, ForeignKey('users.id'))
    locdatetime = Column(DateTime)
    locid = Column(Integer, ForeignKey('locations.id'))

    user = relationship('User', back_populates='checkins')
    location = relationship('Location', back_populates='checkins')


class CSV:
    chunksize = 10 ** 6  # how many rows to read (and then insert into database) each loop
    coltypes = None  # column types
    colmapper = None  # a mapper from csv column to table column
    converters = None  # read_csv()'s converters for certain csv columns
    encoding = None  # code page of csv
    filename = None  # fullpath of csv


class Table:
    insert_prefixes = None  # Add one or more expressions following INSERT command, usually for conflict resolution.
    table = None  # SQLAlchemy Table object


class CSV2SQLAlchemy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.engine = create_engine(dataset.url)
        Base.metadata.create_all(self.engine)
        scopedsession = scoped_session(sessionmaker(bind=self.engine))
        self.session = scopedsession()

    def __del__(self):
        self.session.close()

    def _chunk_insert(self, table, csv):
        chunk_cnt = 0
        for chunk in read_csv(csv.filename, chunksize=csv.chunksize, dtype=csv.coltypes,
                              converters=csv.converters, encoding=csv.encoding):
            print(chunk_cnt)
            chunk_rows = [dict(zip(csv.colmapper, row)) for row in chunk.values.tolist()]
            self.engine.execute((table.table.insert(prefixes=table.insert_prefixes)), chunk_rows)
            chunk_cnt += 1

    @classmethod
    def str2datetime(cls, datetime_str):
        return parse(datetime_str)

    @classmethod
    def str2jsonstr(cls, string):
        return string[1:-1].replace("'", '"')

    def _import_users(self):
        table = Table()
        table.table = User.__table__

        csv = CSV()
        csv.filename = self.dataset.csvs['gowalla_userinfo']
        csv.colmapper = ['id', 'bookmarked_spots_count', 'challenge_pin_count', 'country_pin_count',
                         'highlights_count', 'items_count', 'photos_count', 'pins_count', 'province_pin_count',
                         'region_pin_count', 'state_pin_count', 'trips_count', 'friends_count', 'stamps_count',
                         'checkin_num', 'places_num']
        csv.coltypes = np.int

        # set all columns except global_userid
        self._chunk_insert(table, csv)

        # set global_userid = userid for each row
        self.session.query(User).update({'global_userid': User.id})
        self.session.commit()

    def _import_edges(self):
        table = Table()
        table.table = edges

        csv = CSV()
        csv.filename = self.dataset.csvs['gowalla_edges']
        csv.colmapper = ['userid', 'friendid']
        csv.coltypes = {'userid1': np.int32, 'userid2': np.int32}

        self._chunk_insert(table, csv)

    def _import_locations(self):
        table = Table()
        table.table = Location.__table__
        # There are conflicts for INSERT command since there are duplicate ids in gowalla_spots_subset2.csv.
        # Note that : 1) there is no duplicate id in gowalla_spots_subset1.csv, and
        # 2) there is no common id between gowalla_spots_subset1.csv and gowalla_spots_subset2.csv.

        csv1 = CSV()
        csv1.filename = self.dataset.csvs['gowalla_spots_subset1']
        csv1.colmapper = ['id', 'created_at', 'lon', 'lat', 'photos_count', 'checkins_count', 'users_count',
                          'radius_meters', 'highlights_count', 'items_count', 'max_items_count', 'spot_categories']
        csv1.coltypes = {'id': np.int32, 'lng': np.float64, 'lat': np.float64, 'photos_count': np.int32,
                         'checkins_count': np.int32, 'users_count': np.int32, 'radius_meters': np.int32,
                         'highlights_count': np.int32, 'items_count': np.int32, 'max_items_count': np.int32}
        csv1.converters = {'created_at': self.str2datetime, 'spot_categories': self.str2jsonstr}
        self._chunk_insert(table, csv1)

        table.insert_prefixes = ['OR REPLACE']
        csv2 = CSV()
        csv2.filename = self.dataset.csvs['gowalla_spots_subset2']
        csv2.colmapper = ['id', 'lat', 'lon', 'name', 'city_state']
        csv2.coltypes = {'id': np.int32, 'lat': np.float64, 'lng': np.float64, 'name': str, 'city_state': str}
        csv2.encoding = 'utf-8'

        self._chunk_insert(table, csv2)

    def _import_checkins(self):
        table = Table()
        table.table = Checkin.__table__

        csv = CSV()
        csv.filename = self.dataset.csvs['gowalla_checkins']
        csv.colmapper = ['userid', 'locid', 'locdatetime']
        csv.coltypes = {'userid': np.int32, 'placeid': np.int32}
        csv.converters = {'datetime': self.str2datetime}

        self._chunk_insert(table, csv)

    def clear(self, tables=None):
        if tables:
            for table in tables:
                self.session.execute(table.delete())
            self.session.commit()

    def clear_all(self):
        self.clear(reversed(Base.metadata.sorted_tables))
        # or:
        # Base.metadata.reflect(bind=self.engine)
        # Base.metadata.clear()

    def import_all(self):
        self._import_users()
        self._import_edges()
        self._import_locations()
        self._import_checkins()


if __name__ == '__main__':
    importer = CSV2SQLAlchemy(dataset=GowallaSQLAlchemy)
    importer.clear()
    # importer.import_all()
