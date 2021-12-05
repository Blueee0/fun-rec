import sys

from sqlalchemy.sql.functions import user
sys.path.append('../../')

from dao.mongo_server import MongoServer
from dao.redis_server import RedisServer
from dao.mysql_server import MysqlServer
from dao.entity.register_user import RegisterUser
from datetime import datetime
import random

# 这里需要从物料库中获取物料的信息，把物料按冷启用户分组
# 对不同组的冷启用户推荐对应物料50条 + 10条本地新闻 按热度值排序 并去重 

"""
第一版先不考虑用户位置和新闻位置的关系，因为目前新闻的数据量太少了
冷启动用户分组，最终的内容按照热度进行排序：
"""

class ColdStart(object):
    def __init__(self):
        self.feature_protrail_collection = MongoServer().get_feature_protrail_collection()
        self.reclist_redis = RedisServer().get_reclist_redis()
        self.register_user_sess = MysqlServer().get_register_user_session()
        self.set_user_group()

    def set_user_group(self):
        """将用户进行分组
        1. age < 23 && gender == female  
        2. age >= 23 && gender == female 
        3. age < 23 && gender == male 
        4. age >= 23 && gender == male  
        """
        self.user_group = {
            "1": ["国内","娱乐","体育","科技"],
            "2": ["国内","社会","美股","财经","股市"],
            "3": ["国内","股市","体育","科技"],
            "4": ["国际", "国内","军事","社会","美股","财经","股市"]
        }

    def generate_cold_user_strategy_templete_to_redis(self):
        """冷启动用户模板，总共分成了四类人群
        """
        for k, item in self.user_group.items():
            redis_key = "cold_start_group:{}".format(str(k))
            query_keys_list = [{"cate": val} for val in item]
            news_info_list = list(self.feature_protrail_collection.find({'$or': query_keys_list}))
            # 为了后面拿到文章的类别
            news_id_list = [news["cate"] + "_" + news["news_id"] for news in news_info_list]
            hot_values_list = [float(news['hot_value']) for news in news_info_list]
            print("模板 {} 的 新闻数量为 {}".format(str(k), len(news_info_list)))
            for news, score in zip(news_id_list, hot_values_list):
                self.reclist_redis.zadd(redis_key, {news: score}, nx=True)

    def copy_redis_sorted_set(self, user_id, redis_key):
        """redis冷启动模板拷贝到当前用户，由于需要产生一定的差异，这里随机对模板中的新闻随机选择90%的数据
        TODO: 需要有随机性
        """
        user_key = "cold_start:{}".format(user_id)
        print(self.reclist_redis.zunionstore(user_key, [redis_key]))

    def generate_cold_start_news_list_to_redis_for_register_user(self):
        """给已经注册的用户制作冷启动新闻列表
        """
        for user_info in self.register_user_sess.query(RegisterUser).all():
            if int(user_info.age) < 23 and user_info.gender == "female":
                redis_key = "cold_start_group:{}".format(str(1))
                self.copy_redis_sorted_set(user_info.userid, redis_key)
            elif int(user_info.age) >= 23 and user_info.gender == "female":
                redis_key = "cold_start_group:{}".format(str(2))
                self.copy_redis_sorted_set(user_info.userid, redis_key)
            elif int(user_info.age) < 23 and user_info.gender == "male":
                redis_key = "cold_start_group:{}".format(str(3))
                self.copy_redis_sorted_set(user_info.userid, redis_key)
            elif int(user_info.age) >= 23 and user_info.gender == "male":
                redis_key = "cold_start_group:{}".format(str(4))
                self.copy_redis_sorted_set(user_info.userid, redis_key)
            else:
                pass 
        print("generate_cold_start_news_list_to_redis_for_register_user.")

if __name__ == "__main__":
    ColdStart().generate_cold_user_strategy_templete_to_redis()
    ColdStart().generate_cold_start_news_list_to_redis_for_register_user()

