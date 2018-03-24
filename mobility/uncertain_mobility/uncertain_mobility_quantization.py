# encoding=utf-8
import numpy as np
from dataset import SNAPGowallaAustinDLSK5M5 as Dadaset
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from utils.db_access.db_access import DBManipulation
from utils.io import save


class UncertainMobilityQuantization:
    INVALID_ANONYMOUS_LOCATIONs = [(-1, -1)]

    def __init__(self, dataset):
        self.db = DBManipulation(dataset)
        self.dataset = dataset
        self.userids = self.get_userids()
        self.locations = self.get_locations()
        self.vfds = None

    def get_userids(self):
        print('get_userids')
        self.db.cursor.execute(''.join(['SELECT DISTINCT userid',
                                        ' FROM ', self.dataset.checkins,
                                        ' ORDER BY userid']))
        userids = {row[0]: idx for idx, row in enumerate(self.db.cursor.fetchall())}
        return userids

    def get_locations(self):
        print('get_locations')
        self.db.cursor.execute(''.join(['SELECT DISTINCT lon0, lat0',
                                        ' FROM ', self.dataset.checkins,
                                        ' ORDER BY lon0, lat0']))
        locations = {(row[0], row[1]): idx for idx, row in enumerate(self.db.cursor.fetchall())}
        return locations

    def visit_frequency_distributions(self):
        print('visit_frequency_distributions')
        distributions = np.zeros(shape=(len(self.userids), len(self.locations)))
        for userid, udseridx in self.userids.items():
            for i in range(self.dataset.k):
                i_str = str(i)
                self.db.cursor.execute(''.join(['SELECT lon', i_str, ', lat', i_str, ', COUNT(*)',
                                                ' FROM ', self.dataset.checkins,
                                                ' WHERE userid = ?',
                                                ' GROUP BY lon', i_str, ', lat', i_str,
                                                ' ORDER BY lon', i_str, ', lat', i_str]), (userid,))
                for lon, lat, cnt in self.db.cursor.fetchall():
                    distributions[udseridx, self.locations[(lon, lat)]] += cnt
        self.vfds = normalize(distributions, copy=False)

    def visit_frequency_cosine_similarities(self):
        print('visit_frequency_cosine_similarities')
        if self.vfds is None:
            self.visit_frequency_distributions()
        similarities = np.ones(shape=(len(self.userids), len(self.userids)))
        for i in range(len(self.userids)):
            for j in range(i + 1, len(self.userids)):
                similarities[i, j] = cosine(self.vfds[i, :], self.vfds[j, :])
                similarities[j, i] = similarities[i, j]
        return similarities


if __name__ == '__main__':
    umq = UncertainMobilityQuantization(dataset=Dadaset)
    vfcs = umq.visit_frequency_cosine_similarities()
    save(vfcs, '%d_anonymous_visit_frequency_cosine_similarities.dat' % umq.dataset.k)
