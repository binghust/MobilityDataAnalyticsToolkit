# encoding=utf-8
from db_manipulation import DBManipulation


class SocialTiesQuantization:
    def __init__(self, dataset_name='Brightkite', social_ties_table_name='Edges'):
        self.db = DBManipulation(dataset_name)
        self.tb_name = social_ties_table_name
        self.userids = self.get_userids()

    def get_userids(self):
        sql_select = ''.join(['SELECT DISTINCT userid FROM ', self.tb_name])
        self.db.cursor.execute(sql_select)
        userids = [row[0] for row in self.db.cursor.fetchall()]
        return userids

    def friendids_of(self, userid):
        sql_select = ''.join(['SELECT DISTINCT friendid FROM ', self.tb_name, ' WHERE userid = ?'])
        self.db.cursor.execute(sql_select, (userid,))
        friendids = set([row[0] for row in self.db.cursor.fetchall()])
        return friendids

    def two_degree_friendids_of(self, userid):
        one_degree_friendids = self.friendids_of(userid)
        placeholders = ', '.join(['?'] * len(one_degree_friendids))
        sql_select = ''.join(['SELECT DISTINCT friendid FROM ', self.tb_name, ' WHERE userid IN (', placeholders, ')'])
        two_degree_friendids = [row[0] for row in self.db.cursor.execute(sql_select, one_degree_friendids)]
        return two_degree_friendids

    def three_degree_friendids_of(self, userid):
        two_degree_friendids = self.two_degree_friendids_of(userid)
        placeholders = ', '.join(['?'] * len(two_degree_friendids))
        sql_select = ''.join(['SELECT DISTINCT friendid FROM ', self.tb_name, ' WHERE userid IN (', placeholders, ')'])
        three_degree_friendids = [row[0] for row in self.db.cursor.execute(sql_select, two_degree_friendids)]
        return three_degree_friendids

    def common_friendids_between(self, userid1, userid2):
        friendids1 = set(self.friendids_of(userid1))
        friendids2 = set(self.friendids_of(userid2))
        intersect_of_friendids = friendids1.intersection(friendids2)
        return intersect_of_friendids

    def jaccard_similarity(self, userid1, userid2):
        friendids1 = set(self.friendids_of(userid1))
        friendids2 = set(self.friendids_of(userid2))
        intersect_of_friendids = friendids1.intersection(friendids2)
        union_of_friendids = friendids1.union(friendids2)
        similarity = len(intersect_of_friendids) / len(union_of_friendids)
        return similarity

    def jaccard_similarity_for_all_users(self):
        column_name = 'jaccard1'
        target_table_name = self.tb_name + '_' + column_name
        # create_function(name, num_params, func): sqlite user-defined function,
        # refer to https://docs.python.org/3.6/library/sqlite3.html#sqlite3.Connection.create_function
        self.db.conn.create_function('jaccard_similarity', 2, self.jaccard_similarity)
        sql_create_table = ''.join(['DROP TABLE ', 'IF EXISTS ', target_table_name,
                                    ';CREATE TABLE ', target_table_name, ' AS',
                                    ' SELECT userid, friendid, jaccard_similarity(userid, friendid) as ', column_name,
                                    ' FROM ', self.tb_name])
        self.db.cursor.executescript(sql_create_table)

    def community_detection(self):
        pass


if __name__ == '__main__':
    stq = SocialTiesQuantization(
        dataset_name='Gowalla',
        social_ties_table_name='Edges_Stockholm_Time_Sample3600_Length50_Cluster50'
    )
    stq.jaccard_similarity_for_all_users()
