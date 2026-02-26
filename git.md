# git 相关操作

## 配置SSH linux
```python
    ls ~/.ssh
    #没有则生成
    ssh-keygen -t rsa -b 4096 -C "你的邮箱"
    #复制公钥
    cat ~/.ssh/id_rsa.pub
    #拉取代码
    git clone git@github.com:antushengdequanshijie/note.git
    #配置用户名和邮箱
    git config --global user.name "你的名字"
    git config --global user.email "你的邮箱@example.com"
    #上传代码
    cd ~/GitHub/note          # 进入项目目录
    git status                 # 查看修改
    git add .                  # 添加所有改动
    git commit -m "更新 2026-02-26 Markdown 日志"
    git push -u origin main    # 第一次 push，注意main分支
