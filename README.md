# imgDescribe
Image Description Tool

这是一个轻量工具，使用streamlit和pyvisionai，可利用Ollama本地模型llama 3.2 vision实现图片反推提示词（图片描述）功能，并且用Baidu API实现中文翻译。


## 安装

建议用Conda安装python3.12环境 conda create --name imgtool python=3.12

安装依赖 pip install -r requrements.txt

为用起来方便，请设置Baidu API的两个参数到环境变量：

Windows：
   set BAIDU_APPID=【YOUR ID】
   set BAIDU_APPKEY=【YOUR KEY】

Linux:
   set BAIDU_APPID=【YOUR ID】
   set BAIDU_APPKEY=【YOUR KEY】

## 运行

streamlit run imgDescribe.py


## 示例图片

![示例图片1](interface1.png)

![示例图片2](interface2.png)

![示例图片3](interface3.png)



鸣谢：
1、感谢streamlit和pyvisionai的作者，让几百行python代码就可以实现一个较为实用的功能。
2、感谢Ollama提供了本地Vision LLM的调用.