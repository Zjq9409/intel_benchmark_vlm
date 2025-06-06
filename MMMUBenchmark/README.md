# 运行MMMU测试的准备工作
## 1. 首先导入测试镜像：
```bash
docker load -i mmmu_1.0.tar
```
镜像文件保存在10.238.149.77的/DISK3/home/llm/multimodalLLM/MMMU_Test目录下。
## 2. 启动镜像
```bash
docker run -itd  -v /llm:/llm \
	 --ipc=host \
	 --name mmmu \
	 mmmu:2.0
```
host上的/llm目录需要包含MMMU以及MMMMUPro的测试数据集，这两个数据集可从10.238.149.77的/DISK3/home/llm/multimodalLLM/MMMU_Test目录中下载。
## 2. 启动模型vllm服务
参考vllm启动的脚本启动llm服务，模型上下文最少设置为8K，推荐16K。

## MMMU/MMMU-Pro测试
1. 启动镜像
```bash
docker exec -it mmmu /bin/bash
```
2. 进入测试目录
```bash
cd /mmmu/MMMU/mmmu-pro/infer
```
3. 运行测试
### MMMUPro测试
```bash
 python infer_gpt.py "gemma3-12b" "http://192.168.10.211:8000/v1" cot "standard (10 options)" "/llm/MMMUPro"
 ```

###  MMMU测试
```bash
 python infer_gpt.py "gemma3-12b" "http://192.168.10.211:8000/v1" cot "standard (10 options)" "/llm/MMMU"
 ```
 参数含义如下：
 1. vllm服务名称 - "gemma3-12b" 
 2. vllm服务地址 - "http://192.168.10.211:8000/v1" 
 3. 服务是否开启COT - cot（开启），direct (不开启)
 4. 测试的分类 - "standard (10 options)" 
 5. 数据集所在目录 - "/llm/MMMU"

测试会在目录下生成output文件夹，一共后面生成准确率使用。


4. 计算准确率
```bash
cp -R output ../
cd ..
python evaluate.py
```
输出类似：
```
Model: gemma3-12b      Method: direct   Setting: standard - Accuracy:  44.56%
```
最后的Accuracy就是模型准确率。
