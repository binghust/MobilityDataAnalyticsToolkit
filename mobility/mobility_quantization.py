# encoding=utf-8

import numpy as np
from dataset import GowallaSQLAlchemy as Dataset
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from sqlalchemy import create_engine, func
from sqlalchemy.orm import scoped_session, sessionmaker
from utils.db_access.sqlite2sqlalchemy import Checkin, Location, User


class MobilityQuantization:
    def __init__(self, dataset):
        self.engine = create_engine(dataset.url)
        self.session = self.get_session()
        self.users = self.session.query(User.id).all()
        self.locations = self.session.query(Location.id).all()
        self.vfds = None

    def get_session(self):
        print('get_session')
        # Base.metadata.create_all(self.engine)
        scopedsession = scoped_session(sessionmaker(bind=self.engine))
        return scopedsession()

    def visit_frequency_distributions(self):
        print('visit_frequency_distributions')
        distributions = np.zeros(shape=(len(self.users), len(self.locations)))
        for user in self.users:
            result = self.session.query(Checkin.locid, func.count(Checkin.locid)). \
                group_by(Checkin.locid).filter(Checkin.userid == user.id).all()
            for locid, cnt in result:
                distributions[user.id, locid] = cnt
        self.vfds = normalize(distributions, copy=False)

    def visit_frequency_cosine_similarities(self):
        print('visit_frequency_cosine_similarities')
        if self.vfds is not None:
            similarities = np.ones(shape=(len(self.users), len(self.users)))
            for i in range(len(self.users)):
                for j in range(i + 1, len(self.users)):
                    similarities[i, j] = cosine(self.vfds[i, :], self.vfds[j, :])
                    similarities[j, i] = similarities[i, j]
            return similarities


def main():
    mq = MobilityQuantization(dataset=Dataset)
    mq.visit_frequency_distributions()
    vfcs = mq.visit_frequency_cosine_similarities()
    # save(vfcs, 'actual_visit_frequency_cosine_similarities.dat')


if __name__ == '__main__':
    main()
