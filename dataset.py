# encoding=utf-8
class Dataset:
    checkins = None
    csvs = None
    edges = None
    name = None
    ranges = None
    url = None


class SNAPBrightkite(Dataset):
    checkins = 'Checkins'
    csvs = {'checkins': 'D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Brightkite\\checkins.txt',
            'edges': 'D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Brightkite\\edges.txt'}
    edges = 'Edges'
    name = 'SNAPBrightkite'
    ranges = [-180.0, 180.0, -80.0, 80.0]
    url = 'D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Brightkite\\checkins.db'


class SNAPBrightkitePreview(SNAPBrightkite):
    checkins = 'Checkins_Preview'
    edges = 'Checkins_Preview'
    name = 'SNAPBrightkite_Preview'


class SNAPGowalla(Dataset):
    checkins = 'Checkins'
    csvs = {'checkins': 'D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Gowalla\\checkins.txt',
            'edges': 'D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Gowalla\\edges.txt'}
    edges = 'Edges'
    name = 'SNAPGowalla'
    ranges = [-180.0, 180.0, -80.0, 80.0]
    url = 'D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Gowalla\\checkins.db'


class SNAPGowallaAustin(SNAPGowalla):
    checkins = 'Checkins_Austin_Time_Sample3600_Length100_Cluster50'
    edges = 'Edges_Austin_Time_Sample3600_Length100_Cluster50'
    name = 'SNAPGowalla_Austin'
    ranges = [-97.7714033167, -97.5977249833, 30.19719445, 30.4448463144]


class SNAPGowallaAudtinSQLAlchemy(SNAPGowallaAustin):
    checkins = 'checkins'
    edges = 'edges'
    name = 'SNAPGowalla_Audtin_SQLAlchemy'
    url = 'sqlite:///D:\\Workspace\\Datasets\\Location-Based Social Network\\SNAP Gowalla\\checkins_sqlalchemy.db'


class SNAPGowallaPreview(SNAPGowalla):
    checkins = 'Checkins_Preview'
    edges = 'Edges_Preview'
    name = 'SNAPGowalla_Preview'


class SNAPGowallaStockholm(SNAPGowalla):
    checkins = 'Checkins_Stockholm_Time_Sample3600_Length100_Cluster50'
    edges = 'Edges_Stockholm_Time_Sample3600_Length100_Cluster50'
    name = 'SNAPGowalla_Stockholm'
    ranges = [17.911736377, 18.088630197, 59.1932443, 59.4409599167]


class SNAPGowallaSanfrancisco(SNAPGowalla):
    checkins = 'Checkins_SF_Time_Sample3600_Length100_Cluster50'
    edges = 'Edges_SF_Time_Sample3600_Length100_Cluster50'
    name = 'SNAPGowalla_Sanfrancisco'
    ranges = [-122.521368, -122.356684, 37.706357, 37.817344]


class SNAPGowallaAustinDLSK5M5(SNAPGowallaAustin):
    k = 5
    m = 5
    checkins = 'Checkins_Stockholm_Time_Sample3600_Length50_Cluster50_DLS_K5_M5'
    edges = 'Edges_Stockholm_Time_Sample3600_Length100_Cluster50'
    name = 'SNAPGowalla_Austin_KAnonymous_DLS_K5_M5'


class GowallaSQLAlchemy(Dataset):
    checkins = 'checkins'
    csvs = {'gowalla_checkins': 'D:\\Workspace\\Datasets\\Location-Based Social Network\\Gowalla\\gowalla_checkins.csv',
            'gowalla_edges': 'D:\\Workspace\\Datasets\\Location-Based Social Network\\Gowalla\\gowalla_friendship.csv',
            'gowalla_userinfo': 'D:\\Workspace\\Datasets\\Location-Based Social Network\\Gowalla\\gowalla_userinfo.csv',
            'gowalla_spots_subset1':
                'D:\\Workspace\\Datasets\\Location-Based Social Network\\Gowalla\\gowalla_spots_subset1.csv',
            'gowalla_spots_subset2':
                'D:\\Workspace\\Datasets\\Location-Based Social Network\\Gowalla\\gowalla_spots_subset2.csv'}
    edges = 'edges'
    name = 'GowallaSQLAlchemy'
    ranges = [-180.0, 180.0, -80.0, 80.0]
    url = 'sqlite:///D:\\Workspace\\Datasets\\Location-Based Social Network\\Gowalla\\checkins.db'
