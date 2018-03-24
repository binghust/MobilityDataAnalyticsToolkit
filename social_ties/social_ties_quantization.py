# encoding=utf-8
from pickle import dump

import numpy as np
from dataset import SNAPGowallaStockholm as Dataset
from utils.db_access.db_access import DBManipulation


class SocialTiesQuantization:
    def __init__(self, dataset):
        self.db = DBManipulation(dataset)
        self.edges = dataset.edges
        self.userids = self.get_userids()

    def get_userids(self):
        print('get_userids')
        sql_select = ''.join(['SELECT DISTINCT userid FROM ', self.edges])
        self.db.cursor.execute(sql_select)
        userids = [row[0] for row in self.db.cursor.fetchall()]
        return userids

    def friendids_of(self, userid):
        print('friendids_of')
        sql_select = ''.join(['SELECT DISTINCT friendid FROM ', self.edges, ' WHERE userid = ?'])
        self.db.cursor.execute(sql_select, (userid,))
        friendids = set([row[0] for row in self.db.cursor.fetchall()])
        return friendids

    def two_degree_friendids_of(self, userid):
        print('two_degree_friendids_of')
        one_degree_friendids = self.friendids_of(userid)
        placeholders = ', '.join(['?'] * len(one_degree_friendids))
        sql_select = ''.join(['SELECT DISTINCT friendid FROM ', self.edges, ' WHERE userid IN (', placeholders, ')'])
        two_degree_friendids = [row[0] for row in self.db.cursor.execute(sql_select, one_degree_friendids)]
        return two_degree_friendids

    def three_degree_friendids_of(self, userid):
        print('three_degree_friendids_of')
        two_degree_friendids = self.two_degree_friendids_of(userid)
        placeholders = ', '.join(['?'] * len(two_degree_friendids))
        sql_select = ''.join(['SELECT DISTINCT friendid FROM ', self.edges, ' WHERE userid IN (', placeholders, ')'])
        three_degree_friendids = [row[0] for row in self.db.cursor.execute(sql_select, two_degree_friendids)]
        return three_degree_friendids

    def common_friendids_between(self, userid1, userid2):
        print('common_friendids_between')
        friendids1 = set(self.friendids_of(userid1))
        friendids2 = set(self.friendids_of(userid2))
        intersect_of_friendids = friendids1.intersection(friendids2)
        return intersect_of_friendids

    def jaccard_similarity(self, userid1, userid2):
        print('jaccard_similarity')
        friendids1 = set(self.friendids_of(userid1))
        friendids2 = set(self.friendids_of(userid2))
        intersect_of_friendids = friendids1.intersection(friendids2)
        union_of_friendids = friendids1.union(friendids2)
        similarity = len(intersect_of_friendids) / len(union_of_friendids)
        return similarity

    def jaccard_similarities(self):
        print('jaccard_similarities')
        similarities = np.ones(shape=(len(self.userids), len(self.userids)))
        for i in range(len(self.userids)):
            for j in range(i + 1, len(self.userids)):
                similarities[i, j] = self.jaccard_similarity(self.userids[i], self.userids[j])
                similarities[j, i] = similarities[i, j]
        return similarities


def save(var, filename):
    with open(filename, 'wb') as fh:
        dump(var, fh)


if __name__ == '__main__':
    stq = SocialTiesQuantization(dataset=Dataset)
    js = stq.jaccard_similarities()
    save(js, 'jaccard_similarities.dat')
