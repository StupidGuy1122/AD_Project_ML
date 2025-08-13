About how to test locally:
1.you need to have all the data prepared and dockerweb downloaded.
2.Because there always happens bug while change the document import or something else, in this readme, I will teach
you how to use docker to build an image and use dockerweb to test the api and something else.
3.First, you need to cd in this project's top route
4.type this code on cmd to build an image call adprojectml:local and port"." can't be skipped:
docker build -t adprojectml:local .
5.type this code on cmd to turn on the image and expose it to the 80 port, you can also manually turn on the exact image on the dockerweb:
docker run -p 80:80 adprojectml:local
6.turn on a browser and type this url to open the swagger page, in this page you can test the api without using postman
http://localhost/docs#/
7.I will list every json body input here:
    recommend Activity:
	POST /recommendActivity/
	{
	  "user_id": 1,
	  "top_k": 2
	}
	返回数据为：JSON{
		"user_id": request.user_id,
		"recommended_activity_ids": recommendations#['id', 'title', 'score']
	}

	recommend user:
	POST /recommendUser/
	{
	  "user_id": 1,
	  "top_k": 3
	}
	返回数据为：JSON{
		"user_id": request.user_id,
        "similar_users": [
            {"user_id": uid, "similarity": score} for uid, score in sim_users
        ]
	}

	predict Tags:
    POST /predictTags/
    Content-Type: application/json
    {
      "title": "周末户外徒步活动",
      "description": "穿越森林探索自然美景，适合所有年龄段参与者"
    }
    返回数据：
    {
      "predicted_tags": ["户外", "徒步", "自然", "周末活动"]
    }

    train model:
    GET /TrainRecommender/  or   GET /TrainTagPredictor/
    you can see the trainning log on your dockerweb's cmd
8.Every time you upload the python file, you need to repeate the step 4-6 to update the code.
(option)
this is a small way about how to use completely local way to use the project :
fist change all the dependency import, like from "from app.tagpredictor_def" to "tagpredictor_def"
then change the database connection from using azure connection to the pysql to connect to the local database
third just run the corresponding file to test the function or
      run "uvicorn app.main:app --reload --port 8000" to turn on the swagger







活动与相似用户推荐使用步骤：
1、conda install torch，numpy，pandas，fastapi，scikit-learn, pymysql
2、每当需要更新模型或者首次训练模型时：run python app/trainrecommender.py
3、需要测试模型但不需要使用API时,run app/recommender.py，注意自行在代码中更改推荐top_k个数据等
4、需要调用API时，需要 run uvicorn app.main:app --reload --port 8000
5、API传输json例子：
	recommend Activity：
	POST /recommendActivity/
	{
	  "user_id": 1,
	  "top_k": 2
	}
	返回数据为：JSON{
		"user_id": request.user_id, 
		"recommended_activity_ids": recommendations#['id', 'title', 'score']
	}
	recommend user：
	POST /recommendUser/
	{
	  "user_id": 1,
	  "top_k": 3
	}
	返回数据为：JSON{
		"user_id": request.user_id,
        "similar_users": [
            {"user_id": uid, "similarity": score} for uid, score in sim_users
        ]
	}

接下来是模型的粗略介绍：
预处理部分：
1、首先将每个活动的信息按类型处理整合成一个特征向量：该向量由三部分组成：文本向量（300维）、标签向量（多标签独热编码）、时间特征（3维：hour_sin, hour_cos, duration）
2、然后对用户标签处理：经过多标签编码后构成一个标签索引序列，标签索引序列经过嵌入层和平均池化后形成稠密标签表示
3、将用户id和操作2得到的稠密标签进行嵌入，形成用户特征矩阵
3、将经过一个全连接层处理后的1中的特征向量和对应的活动id拼接起来，形成活动特征矩阵
4、将3中的用户特征矩阵和4中的活动特征矩阵拼接起来
模型预测部分：
将预处理后的混合特征矩阵交给深度协同过滤网络：由三层MLP构成，处理后进行最终预测

训练阶段：使用交互行为（收藏/参加）作为监督信号训练模型
推荐阶段：过滤掉已交互的活动，仅对新活动进行评分与排序

活动tag预测使用教程：
#模型缺陷基本基于tag定义，数据库中的tag涵盖范围不全的同时有交叉覆盖的情况，所以我在训练时使用了部分过拟合的机制，加强对高级tag的注意力
1、conda install torch，numpy，pandas，fastapi，scikit-learn,pymysql, transformer
2、每当需要更新模型或者首次训练模型时：run python app/traintagpredictor.py
3、需要测试模型但不需要使用API时,run app/tagpredictor.py，注意自行在代码中更改推荐数据等
4、需要调用API时，需要 run uvicorn app.main:app --reload --port 8000
5、API传输json例子：
    POST /predictTags/
    Content-Type: application/json
    {
      "title": "周末户外徒步活动",
      "description": "穿越森林探索自然美景，适合所有年龄段参与者"
    }
    返回数据：
    {
      "predicted_tags": ["户外", "徒步", "自然", "周末活动"]
    }

另外两个测试训练函数实现功能的API接口：
@app.get("/TrainRecommender/")
@app.get("/TrainTagPredictor/")
任何输出可在docker log里面查看